import argparse, hashlib, hmac, json, math, os, time, requests, torch
import numpy as np, pandas as pd
from urllib.parse import urlencode

try:
    import websocket
except ImportError:
    websocket = None

from dt_model import DecisionTransformer
from dt_utils import compute_step_reward, update_rtg
from backtest import compute_metrics
from features import build_features
from utils import annualization_factor, ensure_dir, load_config, save_json


# --- 辅助函数 ---
def to_act_idx(x):
    return x + 1


def from_act_idx(x):
    return x - 1


def get_ms(iv):
    return int(iv[:-1]) * {"m": 60, "h": 3600, "d": 86400}[iv[-1]] * 1000


def round_step_size(quantity, step_size):
    if step_size <= 0:
        return quantity
    if quantity < step_size:
        return 0.0
    return math.floor(quantity / step_size) * step_size


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    c = ckpt["model_config"]
    m = DecisionTransformer(
        c["state_dim"],
        c["act_dim"],
        c["seq_len"],
        c["d_model"],
        c["n_layers"],
        c["n_heads"],
        c["dropout"],
        action_mode=c.get("action_mode", "discrete"),
        use_value_head=c.get("use_value_head", False),
    ).to(device)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m


def get_lookback(cfg):
    f = cfg["features"]
    wins = f.get("ema_windows", []) + [
        f.get("volatility_window", 0),
        f.get("rsi_window", 0),
        f.get("volume_z_window", 0),
        f.get("zscore_window", 0),
        f.get("boll_window", 0),
        f.get("macd_slow", 0),
    ]
    return int(cfg["dataset"]["seq_len"]) + max(wins) + int(f["zscore_window"]) + 5


def log_csv(path, row):
    ensure_dir(os.path.dirname(path))
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )


def load_logs(path, n=0):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.tail(n).reset_index(drop=True) if n else df


# --- 核心类 ---
class RollingBuffer:
    def __init__(self, df, max_len):
        self.max, self.df = max_len, df.drop_duplicates("timestamp").sort_values(
            "timestamp"
        ).reset_index(drop=True)
        if max_len and len(self.df) > max_len:
            self.df = self.df.tail(max_len).reset_index(drop=True)

    def update(self, bar):
        ts = int(bar["timestamp"])
        if not self.df.empty and self.df.iloc[-1]["timestamp"] == ts:
            for k, v in bar.items():
                self.df.at[self.df.index[-1], k] = v
        else:
            self.df = pd.concat([self.df, pd.DataFrame([bar])], ignore_index=True)
        self.df["datetime"] = pd.to_datetime(self.df["timestamp"], unit="ms", utc=True)
        if self.max and len(self.df) > self.max:
            self.df = self.df.tail(self.max).reset_index(drop=True)


class BinanceClient:
    def __init__(self, cfg):
        self.base = cfg["papertrade"]["base_url"].rstrip("/")
        self.key, self.sec = os.getenv(cfg["papertrade"]["api_key_env"]), os.getenv(
            cfg["papertrade"]["api_secret_env"]
        )
        self._step_size_cache = {}

    def req(self, method, path, params=None, signed=False):
        p = params or {}
        if signed:
            p.update({"timestamp": int(time.time() * 1000), "recvWindow": 5000})
            p["signature"] = hmac.new(
                self.sec.encode(), urlencode(p).encode(), hashlib.sha256
            ).hexdigest()
        r = requests.request(
            method,
            f"{self.base}{path}",
            params=p,
            headers={"X-MBX-APIKEY": self.key} if self.key else {},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def klines(self, sym, iv, limit):
        d = self.req(
            "GET", "/fapi/v1/klines", {"symbol": sym, "interval": iv, "limit": limit}
        )
        df = pd.DataFrame(
            [
                {
                    "timestamp": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                }
                for r in d
            ]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def pos(self, sym):
        return next(
            (
                float(x["positionAmt"])
                for x in self.req("GET", "/fapi/v2/positionRisk", signed=True)
                if x["symbol"] == sym
            ),
            0.0,
        )

    def get_step_size(self, sym):
        if sym in self._step_size_cache:
            return self._step_size_cache[sym]
        info = self.req("GET", "/fapi/v1/exchangeInfo", {"symbol": sym})
        symbols = info.get("symbols", [])
        if not symbols:
            return None
        filters = symbols[0].get("filters", [])
        step_size = None
        for filter_type in ("LOT_SIZE", "MARKET_LOT_SIZE"):
            for f in filters:
                if f.get("filterType") == filter_type and f.get("stepSize") is not None:
                    step_size = float(f["stepSize"])
                    break
            if step_size is not None:
                break
        if step_size is None:
            return None
        self._step_size_cache[sym] = step_size
        return step_size

    def order(self, sym, side, qty):
        return self.req(
            "POST",
            "/fapi/v1/order",
            {"symbol": sym, "side": side, "type": "MARKET", "quantity": qty},
            signed=True,
        )


# --- 监控与风控 ---
ALERT_CACHE = {}


def monitor(cfg, log_path, state, cols, row, logs):
    mc = cfg.get("papertrade", {}).get("monitoring", {})
    if not mc.get("enabled", False):
        return
    ts, cd = row.get("timestamp", int(time.time() * 1000)), float(
        mc.get("cooldown_seconds", 0)
    )

    def emit(type, det):
        if ts - ALERT_CACHE.get(type, 0) >= cd * 1000:
            ALERT_CACHE[type] = ts
            with open(
                mc.get("alert_log_path") or log_path.replace(".csv", ".jsonl"), "a"
            ) as f:
                f.write(
                    json.dumps({"timestamp": ts, "type": type, "details": det}) + "\n"
                )

    err, status = str(row.get("error", "")).strip(), str(row.get("status", "")).upper()
    if err:
        emit("order_error", {"error": err})
    if status == "ERROR":
        emit("order_error_status", {"status": status})

    t_qty, p_qty = float(row.get("target_qty", 0)), float(row.get("position_qty", 0))
    if abs(t_qty - p_qty) > float(mc.get("position_tolerance", 0)) and not err:
        emit("position_mismatch", {"target": t_qty, "actual": p_qty})

    df = pd.concat(
        [logs or pd.DataFrame(), pd.DataFrame([row])], ignore_index=True
    ).tail(int(mc.get("window_bars", 256)))
    if len(df) > 1 and "equity" in df:
        met = compute_metrics(
            df["equity"].values,
            np.append([0], df["equity"].pct_change().dropna().values),
            0,
            0,
            annualization_factor(cfg["papertrade"]["interval"]),
        )
        for k, v in {
            "sharpe": "min_sharpe",
            "total_return": "min_total_return",
            "win_rate": "min_win_rate",
        }.items():
            if (th := mc.get(v)) is not None and met.get(k, 0) < float(th):
                emit(f"degrade_{k}", {k: met[k]})
        if (md := mc.get("max_drawdown_pct")) and met["max_drawdown"] < -abs(float(md)):
            emit("degrade_drawdown", {"drawdown": met["max_drawdown"]})

    # Data Drift
    if th := mc.get("drift_zscore_threshold"):
        base_p = mc.get("baseline_path") or log_path.replace(".csv", "_base.json")
        try:
            base = json.load(open(base_p))
        except:
            base = None
        if not base or base["cols"] != list(cols):
            json.dump(
                {
                    "cols": list(cols),
                    "mean": np.nanmean(state, 0).tolist(),
                    "std": np.nanstd(state, 0).tolist(),
                },
                open(base_p, "w"),
            )
        else:
            z = np.max(
                np.abs(np.nanmean(state, 0) - base["mean"])
                / (np.array(base["std"]) + 1e-8)
            )
            if z > float(th):
                emit("data_drift", {"score": z})


# --- 主逻辑 ---
def run_decision(cfg, model, dev, path, client, df):
    # 1. 特征工程
    sl, fee, slip = (
        int(cfg["dataset"]["seq_len"]),
        float(cfg["rewards"]["fee"]),
        float(cfg["rewards"]["slip"]),
    )
    feat, cols = build_features(df, cfg["features"])
    feat = feat.dropna(subset=cols).reset_index(drop=True)
    if len(feat) < sl:
        return log_csv(
            path,
            {
                "timestamp": int(time.time() * 1000),
                "status": "ERROR",
                "error": "history_too_short",
            },
        )

    # 2. 加载历史上下文 (Logs)
    row = feat.iloc[-1]
    ts, logs = int(row["timestamp"]), load_logs(path, sl + 1)
    if not logs.empty and int(logs.iloc[-1].get("timestamp", 0)) == ts:
        return

    # 3. 构建输入序列
    # 从日志中提取过去的 Action，如果没有则填充 0
    act_mode = getattr(model, "action_mode", "discrete")
    acts = (logs["action"].values if not logs.empty else np.zeros(sl)).astype(
        float if act_mode == "continuous" else int
    )[-sl:]
    p_act = acts[-1] if len(acts) > 0 else 0
    p_eq = (
        float(logs.iloc[-1]["equity"])
        if not logs.empty
        else float(cfg["backtest"]["initial_cash"])
    )

    # 4. 计算当前步奖励 (Current Reward) 用于更新 RTG
    last_r, cur_r = 0.0, 0.0
    if not logs.empty:
        pr = (
            feat[feat["timestamp"] == int(logs.iloc[-1]["timestamp"])].iloc[-1]
            if not logs.empty
            else feat.iloc[-2]
        )
        # 计算上一动作在当前产生的盈亏
        rw_cfg = cfg["rl"]
        cur_r, _, _, _ = compute_step_reward(
            float(p_act),
            float(acts[-2] if len(acts) > 1 else 0),
            float(pr["close"]),
            float(row["close"]),
            fee,
            slip,
            reward_scale=float(rw_cfg.get("reward_scale", 1)),
            turnover_penalty=float(rw_cfg.get("turnover_penalty", 0)),
            position_penalty=float(rw_cfg.get("position_penalty", 0)),
            drawdown_penalty=float(rw_cfg.get("drawdown_penalty", 0)),
            equity=p_eq,
            max_equity=logs["equity"].max() if "equity" in logs else p_eq,
            price_mode=cfg["rewards"].get("price_mode", "close"),
            open_t=float(pr["open"]),
            high_t=float(pr["high"]),
            low_t=float(pr["low"]),
            open_t1=float(row["open"]),
            high_t1=float(row["high"]),
            low_t1=float(row["low"]),
        )

    # 5. RTG (Return-to-Go) 更新
    ds = cfg["dataset"]
    rtg = float(ds.get("target_return", 0))
    for r in logs["reward"].values[-sl:]:
        rtg = update_rtg(rtg, r, float(ds.get("rtg_gamma", 1.0)))
    rtg = update_rtg(rtg, cur_r, float(ds.get("rtg_gamma", 1.0))) / float(
        ds.get("rtg_scale", 1)
    )

    # 6. 模型推理 (Inference)
    # State
    s = torch.tensor(
        feat[cols].tail(sl).to_numpy(dtype=np.float32), device=dev
    ).unsqueeze(0)
    # RTG
    r_in = torch.tensor(np.full(sl, rtg), dtype=torch.float32, device=dev).unsqueeze(0)

    if act_mode == "continuous":
        a_in = torch.tensor(
            np.pad(acts[:-1, None], ((1, 0), (0, 0))), dtype=torch.float32, device=dev
        ).unsqueeze(0)
        logit = model(s, a_in, r_in)
        mod_act = float(torch.tanh(logit[0, -1]).item())
    else:
        a_in = torch.tensor(
            to_act_idx(np.pad(acts[:-1], (1, 0))), dtype=torch.long, device=dev
        ).unsqueeze(0)
        logit = model(s, a_in, r_in)
        mod_act = float(from_act_idx(torch.argmax(logit[0, -1]).item()))

    # 7. 风险控制与熔断 (Kill Switch)
    dry, sym = cfg["papertrade"]["dry_run"], cfg["papertrade"]["symbol"]
    cur_qty = (
        client.pos(sym)
        if not dry
        else float(logs.iloc[-1]["position_qty"] if not logs.empty else 0)
    )

    # 估算当前权益，检查是否触发最大回撤熔断
    eq_now = p_eq * (
        1.0
        + p_act
        * (
            row["close"]
            / float(
                logs.iloc[-1].get("close", row["close"])
                if not logs.empty
                else row["close"]
            )
            - 1.0
        )
    )
    max_eq = max(eq_now, logs["equity"].max() if not logs.empty else eq_now)
    halt = eq_now < max_eq * (
        1.0 - float(cfg["papertrade"].get("max_drawdown_pct", 1.0))
    )

    fin_act, note = (0.0, "kill_switch") if halt else (np.clip(mod_act, -1, 1), "")

    # 8. 仓位管理 (Sizing)
    sz = cfg["papertrade"]["position_sizing"]
    base = float(sz.get("fixed_qty", 0.001))
    if sz.get("mode") == "fixed_notional":
        base = float(sz["fixed_notional"]) / row["close"]
    elif sz.get("mode") == "equity_fraction":
        base = eq_now * float(sz["equity_fraction"]) / row["close"]
    tgt_qty = fin_act * min(
        max(base, float(sz.get("min_qty", 0))), float(sz.get("max_qty", 1e9))
    )

    # 9. 执行交易 (Execution)
    err, ord_res = "", None
    delta = tgt_qty - cur_qty
    step_size = None
    try:
        step_size = client.get_step_size(sym)
    except Exception:
        step_size = None
    if step_size is None:
        step_size = float(cfg["papertrade"].get("step_size", 0.001) or 0.001)
    qty_to_order = round_step_size(abs(delta), step_size)
    min_qty = float(sz.get("min_qty", 0.0))
    if not dry and qty_to_order >= min_qty and qty_to_order > 0:
        try:
            ord_res = client.order(
                sym, "BUY" if delta > 0 else "SELL", qty_to_order
            )
            cur_qty = client.pos(sym)
        except Exception as e:
            err = f"order_fail: {e}"
    elif dry:
        cur_qty = tgt_qty

    # 10. 记录日志
    l_row = {
        "timestamp": ts,
        "close": float(row["close"]),
        "model_action": mod_act,
        "action": fin_act,
        "target_qty": tgt_qty,
        "position_qty": cur_qty,
        "equity": eq_now,
        "reward": cur_r,
        "rtg": rtg * float(ds.get("rtg_scale", 1)),
        "status": "ERROR" if err else ("HALTED" if halt else "OK"),
        "error": err,
        "dry_run": dry,
    }

    monitor(cfg, path, s.cpu().numpy()[0], cols, l_row, logs)
    log_csv(path, l_row)
    if ord_res:
        save_json(path.replace(".csv", "_ord.json"), ord_res)


def run(cfg, model, dev, path, client, loop=False):
    lb = get_lookback(cfg)
    if loop and cfg["papertrade"].get("use_websocket"):
        if not websocket:
            raise ImportError("pip install websocket-client")
        buf = RollingBuffer(
            client.klines(
                cfg["papertrade"]["symbol"], cfg["papertrade"]["interval"], lb
            ),
            lb,
        )
        url = f"{get_ws_base_url(cfg)}/ws/{cfg['papertrade']['symbol'].lower()}@kline_{cfg['papertrade']['interval']}"

        def get_ws_base_url(cfg):
            return (
                "wss://stream.binancefuture.com"
                if "testnet" in cfg["papertrade"]["base_url"]
                else "wss://fstream.binance.com"
            )

        while True:
            try:
                ws = websocket.create_connection(url, timeout=30)
                while True:
                    msg = ws.recv()
                    if not msg:
                        continue
                    k = json.loads(msg).get("k")
                    if not k or (
                        cfg["papertrade"].get("ws_closed_only", True) and not k.get("x")
                    ):
                        continue
                    buf.update(
                        {
                            "timestamp": int(k["t"]),
                            "open": float(k["o"]),
                            "high": float(k["h"]),
                            "low": float(k["l"]),
                            "close": float(k["c"]),
                            "volume": float(k["v"]),
                        }
                    )
                    run_decision(cfg, model, dev, path, client, buf.df)
            except Exception as e:
                log_csv(
                    path,
                    {
                        "timestamp": int(time.time() * 1000),
                        "status": "ERROR",
                        "error": f"ws_err: {e}",
                    },
                )
                time.sleep(float(cfg["papertrade"].get("ws_reconnect_seconds", 5)))
                try:
                    buf = RollingBuffer(
                        client.klines(
                            cfg["papertrade"]["symbol"],
                            cfg["papertrade"]["interval"],
                            lb,
                        ),
                        lb,
                    )
                except Exception as be:
                    log_csv(path, {"status": "ERROR", "error": f"backfill_err: {be}"})
            finally:
                try:
                    ws.close()
                except:
                    pass
    else:
        while True:
            run_decision(
                cfg,
                model,
                dev,
                path,
                client,
                client.klines(
                    cfg["papertrade"]["symbol"], cfg["papertrade"]["interval"], lb
                ),
            )
            if not loop:
                break
            time.sleep(cfg["papertrade"]["poll_seconds"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--loop", action="store_true")
    a = p.parse_args()
    c = load_config(a.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(
        c,
        load_ckpt(a.ckpt, dev),
        dev,
        c["papertrade"]["log_path"],
        BinanceClient(c),
        a.loop,
    )

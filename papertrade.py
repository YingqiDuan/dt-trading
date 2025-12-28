import argparse
import hashlib
import hmac
import json
import os
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import torch

from dt_model import DecisionTransformer
from dt_utils import compute_step_reward, update_rtg
from features import build_features
from utils import ensure_dir, load_config, save_json


def action_to_index(actions):
    return actions + 1


def index_to_action(index):
    return index - 1


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_config"]
    model = DecisionTransformer(
        state_dim=cfg["state_dim"],
        act_dim=cfg["act_dim"],
        seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
        action_mode=cfg.get("action_mode", "discrete"),
        use_value_head=bool(cfg.get("use_value_head", False)),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    return model


def interval_to_millis(interval):
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    if unit == "d":
        return amount * 86_400_000
    raise ValueError(f"unsupported interval {interval}")


class BinanceFuturesClient:
    def __init__(self, base_url, api_key=None, api_secret=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret

    def _headers(self):
        headers = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    def _sign(self, params):
        if not self.api_secret:
            raise ValueError("api_secret required for signed endpoint")
        query = urlencode(params)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def request(self, method, path, params=None, signed=False):
        params = params or {}
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            params = self._sign(params)
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, params=params, headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_klines(self, symbol, interval, limit):
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = self.request("GET", "/fapi/v1/klines", params=params)
        rows = []
        for row in data:
            rows.append(
                {
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def get_position_qty(self, symbol):
        data = self.request("GET", "/fapi/v2/positionRisk", signed=True)
        for row in data:
            if row.get("symbol") == symbol:
                return float(row.get("positionAmt", 0.0))
        return 0.0

    def place_order(self, symbol, side, qty):
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
            "newOrderRespType": "RESULT",
        }
        return self.request("POST", "/fapi/v1/order", params=params, signed=True)


def load_recent_logs(log_path, max_len):
    if not os.path.exists(log_path):
        return pd.DataFrame()
    df = pd.read_csv(log_path)
    if df.empty:
        return df
    return df.tail(max_len).reset_index(drop=True)


def load_last_log(log_path):
    if not os.path.exists(log_path):
        return None
    df = pd.read_csv(log_path)
    if df.empty:
        return None
    return df.iloc[-1]


def compute_equity(prev, close, target_pos, fee, slip, initial_cash):
    if prev is None:
        return initial_cash
    prev_close = float(prev.get("close", close))
    prev_pos = float(prev.get("target_position", prev.get("position", 0.0)))
    prev_equity = float(prev.get("equity", initial_cash))
    ret = (close / prev_close) - 1.0
    trade_cost = (fee + slip) * abs(target_pos - prev_pos)
    return prev_equity * (1.0 + prev_pos * ret - trade_cost)


def append_log(path, row):
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_cycle(cfg, model, device, log_path):
    api_key = os.getenv(cfg["papertrade"]["api_key_env"], "")
    api_secret = os.getenv(cfg["papertrade"]["api_secret_env"], "")
    client = BinanceFuturesClient(cfg["papertrade"]["base_url"], api_key, api_secret)
    action_mode = getattr(model, "action_mode", "discrete")
    reward_fee = float(cfg["rewards"]["fee"])
    reward_slip = float(cfg["rewards"]["slip"])
    reward_scale = float(cfg.get("rl", {}).get("reward_scale", 1.0))
    turnover_penalty = float(cfg.get("rl", {}).get("turnover_penalty", 0.0))
    position_penalty = float(cfg.get("rl", {}).get("position_penalty", 0.0))
    drawdown_penalty = float(cfg.get("rl", {}).get("drawdown_penalty", 0.0))
    price_mode = cfg["rewards"].get("price_mode", "close")
    range_penalty = float(cfg["rewards"].get("range_penalty", 0.0))

    dataset_cfg = cfg.get("dataset", {})
    target_return = float(dataset_cfg.get("target_return", 0.0))
    rtg_scale = float(dataset_cfg.get("rtg_scale", 1.0))
    if rtg_scale <= 0:
        rtg_scale = 1.0
    gamma = float(dataset_cfg.get("rtg_gamma", cfg.get("rl", {}).get("gamma", 1.0)))

    seq_len = cfg["dataset"]["seq_len"]
    ema_windows = [int(w) for w in cfg["features"].get("ema_windows", [])]
    ema_max = max(ema_windows) if ema_windows else 0
    boll_window = int(cfg["features"].get("boll_window", 0))
    macd_fast = int(cfg["features"].get("macd_fast", 0))
    macd_slow = int(cfg["features"].get("macd_slow", 0))
    macd_signal = int(cfg["features"].get("macd_signal", 0))
    max_window = max(
        cfg["features"]["volatility_window"],
        cfg["features"]["rsi_window"],
        cfg["features"]["volume_z_window"],
        cfg["features"]["zscore_window"],
        ema_max,
        boll_window,
        macd_fast,
        macd_slow,
        macd_signal,
    )
    # Extra history is needed because some features apply a rolling window and then a rolling z-score.
    lookback = seq_len + max_window + cfg["features"]["zscore_window"] + 5

    df = client.get_klines(cfg["papertrade"]["symbol"], cfg["papertrade"]["interval"], lookback)
    feat_df, state_cols = build_features(df, cfg["features"])
    feat_df = feat_df.dropna(subset=state_cols).reset_index(drop=True)

    if len(feat_df) < seq_len:
        append_log(
            log_path,
            {
                "timestamp": int(time.time() * 1000),
                "status": "ERROR",
                "error": "insufficient history for features",
            },
        )
        return

    last_row = feat_df.iloc[-1]
    last_timestamp = int(last_row["timestamp"])
    last_log = load_last_log(log_path)
    if last_log is not None and int(last_log.get("timestamp", 0)) == last_timestamp:
        return

    state_window = feat_df[state_cols].tail(seq_len).to_numpy(dtype=np.float32)
    recent_logs = load_recent_logs(log_path, seq_len + 1)
    if not recent_logs.empty and "action" in recent_logs:
        if action_mode == "continuous":
            action_hist = recent_logs["action"].astype(float).tolist()
        else:
            action_hist = recent_logs["action"].astype(int).tolist()
    else:
        action_hist = []
    reward_hist = (
        recent_logs["reward"].astype(float).tolist()
        if not recent_logs.empty and "reward" in recent_logs
        else []
    )

    if action_mode == "continuous":
        actions_in = np.zeros((seq_len, 1), dtype=np.float32)
        if action_hist:
            hist = action_hist[-(seq_len - 1) :]
            actions_in[-len(hist) :, 0] = hist
            if len(action_hist) >= seq_len:
                actions_in[0, 0] = float(action_hist[-seq_len])
    else:
        actions_in = np.zeros(seq_len, dtype=np.int64)
        if action_hist:
            hist = action_hist[-(seq_len - 1) :]
            actions_in[-len(hist) :] = hist
            if len(action_hist) >= seq_len:
                actions_in[0] = int(action_hist[-seq_len])

    prev_equity = float(last_log.get("equity", cfg["backtest"]["initial_cash"])) if last_log is not None else float(
        cfg["backtest"]["initial_cash"]
    )
    max_equity = prev_equity
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        if not log_df.empty and "equity" in log_df.columns:
            max_equity = max(max_equity, float(log_df["equity"].max()))

    prev_row = None
    if last_log is not None:
        last_ts = int(last_log.get("timestamp", 0))
        match = feat_df[feat_df["timestamp"] == last_ts]
        if not match.empty:
            prev_row = match.iloc[-1]
    if prev_row is None and len(feat_df) >= 2:
        prev_row = feat_df.iloc[-2]

    current_reward = 0.0
    if last_log is not None:
        if prev_row is not None and "close" in prev_row:
            prev_close = float(prev_row["close"])
        else:
            prev_close = float(last_log.get("close", last_row["close"]))
        prev_action = float(last_log.get("action", 0.0))
        prev_prev_action = float(action_hist[-2]) if len(action_hist) >= 2 else 0.0
        open_t = float(prev_row["open"]) if prev_row is not None and "open" in prev_row else None
        high_t = float(prev_row["high"]) if prev_row is not None and "high" in prev_row else None
        low_t = float(prev_row["low"]) if prev_row is not None and "low" in prev_row else None
        open_t1 = float(last_row.get("open")) if "open" in last_row else None
        high_t1 = float(last_row.get("high")) if "high" in last_row else None
        low_t1 = float(last_row.get("low")) if "low" in last_row else None
        current_reward, _, _, _ = compute_step_reward(
            float(prev_action),
            float(prev_prev_action),
            float(prev_close),
            float(last_row["close"]),
            reward_fee,
            reward_slip,
            reward_scale=reward_scale,
            turnover_penalty=turnover_penalty,
            position_penalty=position_penalty,
            drawdown_penalty=drawdown_penalty,
            equity=prev_equity,
            max_equity=max_equity,
            price_mode=price_mode,
            open_t=open_t,
            high_t=high_t,
            low_t=low_t,
            open_t1=open_t1,
            high_t1=high_t1,
            low_t1=low_t1,
            range_penalty=range_penalty,
        )

    rtg_values = [float(target_return)]
    current_rtg = float(target_return)
    for past_reward in reward_hist:
        current_rtg = update_rtg(current_rtg, past_reward, gamma)
        rtg_values.append(current_rtg)
    current_rtg = update_rtg(current_rtg, current_reward, gamma)
    rtg_values.append(current_rtg)

    rtg_window = np.zeros(seq_len, dtype=np.float32)
    tail = rtg_values[-seq_len:]
    rtg_window[-len(tail) :] = tail
    rtg_window = rtg_window / rtg_scale

    with torch.no_grad():
        s = torch.tensor(state_window, device=device).unsqueeze(0)
        r = torch.tensor(rtg_window, device=device).unsqueeze(0)
        if action_mode == "continuous":
            a = torch.tensor(actions_in, device=device).unsqueeze(0)
            logits = model(s, a, r)
            action = float(torch.tanh(logits[0, -1]).cpu().numpy().item())
        else:
            a = torch.tensor(action_to_index(actions_in), device=device).unsqueeze(0)
            logits = model(s, a, r)
            action_idx = int(torch.argmax(logits[0, -1]).item())
            action = int(index_to_action(action_idx))

    features_hash = hashlib.sha256(state_window[-1].tobytes()).hexdigest()
    dry_run = cfg["papertrade"]["dry_run"]
    trade_qty = cfg["papertrade"]["trade_qty"]
    target_qty = action * trade_qty

    prev_log = last_log
    equity = compute_equity(
        prev_log,
        float(last_row["close"]),
        float(action),
        cfg["rewards"]["fee"],
        cfg["rewards"]["slip"],
        cfg["backtest"]["initial_cash"],
    )

    peak_equity = max(max_equity, equity)

    max_dd = cfg["papertrade"]["max_drawdown_pct"]
    halted = peak_equity > 0 and equity < peak_equity * (1.0 - max_dd)

    current_qty = 0.0
    order_resp = None
    order_side = ""
    error = ""

    if dry_run:
        if last_log is not None:
            current_qty = float(last_log.get("position_qty", 0.0))
    else:
        try:
            current_qty = client.get_position_qty(cfg["papertrade"]["symbol"])
        except Exception as exc:
            error = f"position_fetch_failed: {exc}"

    if halted and not error:
        error = "kill_switch_triggered"

    delta = target_qty - current_qty
    if not error:
        if abs(delta) > 0:
            order_side = "BUY" if delta > 0 else "SELL"
            if dry_run:
                current_qty = target_qty
            else:
                try:
                    order_resp = client.place_order(cfg["papertrade"]["symbol"], order_side, abs(delta))
                    current_qty = client.get_position_qty(cfg["papertrade"]["symbol"])
                except Exception as exc:
                    error = f"order_failed: {exc}"

    status = "HALTED" if halted else ("ERROR" if error else "OK")
    log_row = {
        "timestamp": last_timestamp,
        "close": float(last_row["close"]),
        "action": action,
        "target_position": action,
        "position": action,
        "position_qty": float(current_qty),
        "order_side": order_side,
        "equity": equity,
        "features_hash": features_hash,
        "logits": json.dumps(logits[0, -1].tolist()),
        "status": status,
        "error": error,
        "dry_run": dry_run,
    }
    log_row["reward"] = float(current_reward)
    log_row["rtg"] = float(current_rtg)

    append_log(log_path, log_row)

    if order_resp is not None:
        resp_path = log_path.replace("trade_log.csv", "last_order.json")
        save_json(resp_path, order_resp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.ckpt).to(device)

    log_path = cfg["papertrade"]["log_path"]
    ensure_dir(os.path.dirname(log_path))

    if args.loop:
        while True:
            run_cycle(cfg, model, device, log_path)
            time.sleep(cfg["papertrade"]["poll_seconds"])
    else:
        run_cycle(cfg, model, device, log_path)


if __name__ == "__main__":
    main()

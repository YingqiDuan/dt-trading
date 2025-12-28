# DT Trading MVP

Minimal, end-to-end pipeline for: data -> features -> Decision Transformer (offline RTG-conditioned SL) -> backtest -> paper trade.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1) Fetch data + build features (optional, caches OHLCV + features)

```bash
python dataset_builder.py --config config.yaml --force
```

2) Train (offline supervised DT on historical trajectories)

```bash
python train_dt.py --config config.yaml
```

Warm-start from saved weights (optimizer resets):

```bash
python train_dt.py --config config.yaml --init_ckpt outputs/checkpoints/dt_best_*.pt
```

3) Backtest (uses best checkpoint by default)

```bash
python backtest.py --config config.yaml
```

4) Paper trade (dry-run by default)

```bash
python papertrade.py --config config.yaml --ckpt outputs/checkpoints/dt_best_*.pt --loop
```

5) Walk-forward (optional)

```bash
# enable walk_forward.enabled in config.yaml first
python walk_forward.py --config config.yaml
```

## Project Structure

- `dataset_builder.py`: fetch OHLCV and build feature CSVs (optional for caching)
- `train_dt.py`: train Decision Transformer policy offline with RTG conditioning
- `backtest.py`: vectorized backtest with friction + baselines
- `papertrade.py`: Binance USDT-M futures paper trading loop
- `walk_forward.py`: rolling train/val/test training + evaluation (offline DT)
- `dt_model.py`: transformer model
- `market_env.py`: step-by-step historical market environment (legacy RL)
- `features.py`: feature engineering
- `config.yaml`: end-to-end config

## Outputs

- `data/raw/`: raw OHLCV
- `data/features/`: feature CSV
- `outputs/train/`: training logs
- `outputs/checkpoints/`: model checkpoints
- `outputs/backtest/`: metrics + equity curve
- `outputs/papertrade/`: trade logs
- `outputs/walk_forward/`: per-fold checkpoints, metrics, and summary

## Key Notes

- RTG (return-to-go) is conditioned during inference; `dataset.target_return` sets the initial prompt.
- Window start uses previous action to avoid transaction-cost bias at sequence boundaries.
- Paper trading uses mainnet URL by default (`https://fapi.binance.com`). Keep `papertrade.dry_run: true` unless you want real orders.
- Metrics include action distribution in `outputs/backtest/metrics.json`.

## Config Tips

- `rl.action_mode`: `discrete | continuous` (continuous uses tanh-squashed Gaussian, action_dim=1).
- `dataset.behavior_policy`: offline action source (`ema_trend|random|buy_hold|flat`).
- `dataset.target_return`, `dataset.rtg_scale`, `dataset.rtg_gamma`: RTG conditioning controls at inference time.
- `rewards.price_mode`, `rewards.range_penalty`: bar pricing and volatility penalty for reward calculation.
- `papertrade.base_url`: `https://fapi.binance.com | https://demo-fapi.binance.com`.
- `walk_forward.*`: rolling window sizes in bars; outputs to `outputs/walk_forward/`.

## Environment Variables

Required for live/testnet orders (not needed for dry-run):

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
```

## Safety

- Start with `papertrade.dry_run: true`.
- Check `outputs/papertrade/trade_log.csv` for errors and position changes.

# DT Trading MVP

Minimal, end-to-end pipeline for: data -> features -> trajectories -> Decision Transformer -> backtest -> paper trade.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1) Build dataset (fetch + features + trajectories)

```bash
python dataset_builder.py --config config.yaml --force
```

2) Train

```bash
python train_dt.py --config config.yaml
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

- `dataset_builder.py`: fetch OHLCV, build features, generate trajectories
- `train_dt.py`: train Decision Transformer (class-weighted CE, macro-F1/recall)
- `backtest.py`: vectorized backtest with friction + baselines
- `papertrade.py`: Binance USDT-M futures paper trading loop
- `walk_forward.py`: rolling train/val/test training + evaluation
- `dt_model.py`: transformer model
- `features.py`: feature engineering
- `config.yaml`: end-to-end config

## Outputs

- `data/raw/`: raw OHLCV
- `data/features/`: feature CSV
- `data/dataset/`: `.npz` trajectory datasets
- `outputs/train/`: training logs
- `outputs/checkpoints/`: model checkpoints
- `outputs/backtest/`: metrics + equity curve
- `outputs/papertrade/`: trade logs
- `outputs/walk_forward/`: per-fold checkpoints, metrics, and summary

## Key Notes

- Return-to-go is conditioned during inference; see `backtest.inference_rtg` and `rtg_quantile_scope` in `config.yaml`.
- Window start uses previous action to avoid transaction-cost bias at sequence boundaries.
- Paper trading uses mainnet URL by default (`https://fapi.binance.com`). Keep `papertrade.dry_run: true` unless you want real orders.
- Metrics include action distribution in `outputs/backtest/metrics.json`.

## Config Tips

- `train.sampling`: `uniform | rtg | episode_return` for weighted sequence sampling.
- `train.use_class_weights`: `true | false` to toggle class-weighted loss.
- `train.use_sampling`: `true | false` to toggle weighted sampling.
- `backtest.inference_rtg`: set to a float for fixed target return; `"auto"` uses a RTG quantile.
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

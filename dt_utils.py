import numpy as np


def resolve_price(price_mode, close, open_=None, high=None, low=None):
    mode = str(price_mode or "close").lower()
    if mode == "close":
        return close
    if mode == "open":
        return close if open_ is None else open_
    if mode in ("hl2", "hl"):
        if high is None or low is None:
            return close
        return 0.5 * (high + low)
    if mode == "oc2":
        if open_ is None:
            return close
        return 0.5 * (open_ + close)
    if mode == "ohlc4":
        if open_ is None or high is None or low is None:
            return close
        return 0.25 * (open_ + high + low + close)
    if mode == "typical":
        if high is None or low is None:
            return close
        return (high + low + close) / 3.0
    return close


def compute_step_reward(
    action,
    prev_action,
    close_t,
    close_t1,
    fee,
    slip,
    reward_scale=1.0,
    turnover_penalty=0.0,
    position_penalty=0.0,
    drawdown_penalty=0.0,
    equity=1.0,
    max_equity=1.0,
    price_mode="close",
    open_t=None,
    high_t=None,
    low_t=None,
    open_t1=None,
    high_t1=None,
    low_t1=None,
    range_penalty=0.0,
):
    price_t = resolve_price(price_mode, close_t, open_=open_t, high=high_t, low=low_t)
    price_t1 = resolve_price(price_mode, close_t1, open_=open_t1, high=high_t1, low=low_t1)
    if price_t == 0:
        ret = 0.0
    else:
        ret = price_t1 / price_t - 1.0
    delta = action - prev_action
    trade_cost = (fee + slip) * abs(delta)
    step_return = action * ret - trade_cost
    reward = step_return
    if turnover_penalty:
        reward -= turnover_penalty * abs(delta)
    if position_penalty:
        reward -= position_penalty * abs(action)
    if range_penalty and high_t1 is not None and low_t1 is not None:
        base = price_t1 if price_t1 != 0 else close_t1
        if base:
            range_ratio = (high_t1 - low_t1) / abs(base)
            reward -= range_penalty * abs(action) * range_ratio

    equity *= 1.0 + step_return
    if equity > max_equity:
        max_equity = equity
    if drawdown_penalty:
        drawdown = max(0.0, (max_equity - equity) / max_equity)
        reward -= drawdown_penalty * drawdown

    reward *= reward_scale
    return float(reward), float(step_return), float(equity), float(max_equity)


def compute_trajectory_rewards(
    actions,
    close,
    fee,
    slip,
    reward_scale=1.0,
    turnover_penalty=0.0,
    position_penalty=0.0,
    drawdown_penalty=0.0,
    open_=None,
    high=None,
    low=None,
    price_mode="close",
    range_penalty=0.0,
):
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)
    close = np.asarray(close, dtype=np.float32).reshape(-1)
    open_ = np.asarray(open_, dtype=np.float32).reshape(-1) if open_ is not None else None
    high = np.asarray(high, dtype=np.float32).reshape(-1) if high is not None else None
    low = np.asarray(low, dtype=np.float32).reshape(-1) if low is not None else None
    rewards = np.zeros(len(actions), dtype=np.float32)
    if len(actions) < 2:
        return rewards

    equity = 1.0
    max_equity = 1.0
    prev_action = 0.0
    for idx in range(len(actions) - 1):
        reward, _, equity, max_equity = compute_step_reward(
            float(actions[idx]),
            prev_action,
            float(close[idx]),
            float(close[idx + 1]),
            fee,
            slip,
            reward_scale=reward_scale,
            turnover_penalty=turnover_penalty,
            position_penalty=position_penalty,
            drawdown_penalty=drawdown_penalty,
            equity=equity,
            max_equity=max_equity,
            price_mode=price_mode,
            open_t=float(open_[idx]) if open_ is not None else None,
            high_t=float(high[idx]) if high is not None else None,
            low_t=float(low[idx]) if low is not None else None,
            open_t1=float(open_[idx + 1]) if open_ is not None else None,
            high_t1=float(high[idx + 1]) if high is not None else None,
            low_t1=float(low[idx + 1]) if low is not None else None,
            range_penalty=range_penalty,
        )
        rewards[idx] = reward
        prev_action = float(actions[idx])
    return rewards


def compute_rtg(rewards, gamma=1.0):
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in reversed(range(len(rewards))):
        running = float(rewards[idx]) + gamma * running
        rtg[idx] = running
    return rtg


def update_rtg(current_rtg, reward, gamma=1.0):
    if gamma == 1.0:
        return float(current_rtg - reward)
    if gamma == 0.0:
        return 0.0
    return float((current_rtg - reward) / gamma)

import numpy as np


def resolve_price(mode, close, open_=None, high=None, low=None):
    mode = str(mode or "close").lower()
    if mode == "open":
        return open_ if open_ is not None else close
    if mode in ("hl2", "hl") and high is not None and low is not None:
        return 0.5 * (high + low)
    if mode == "oc2" and open_ is not None:
        return 0.5 * (open_ + close)
    if mode == "ohlc4" and all(x is not None for x in (open_, high, low)):
        return 0.25 * (open_ + high + low + close)
    if mode == "typical" and high is not None and low is not None:
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
    p_t = resolve_price(price_mode, close_t, open_t, high_t, low_t)
    p_t1 = resolve_price(price_mode, close_t1, open_t1, high_t1, low_t1)
    ret = (p_t1 / p_t - 1.0) if p_t != 0 else 0.0

    delta = action - prev_action
    step_return = action * ret - (fee + slip) * abs(delta)

    reward = step_return
    if turnover_penalty:
        reward -= turnover_penalty * abs(delta)
    if position_penalty:
        reward -= position_penalty * abs(action)
    if range_penalty and high_t1 is not None and low_t1 is not None:
        base = p_t1 if p_t1 != 0 else close_t1
        if base:
            reward -= range_penalty * abs(action) * ((high_t1 - low_t1) / abs(base))

    equity *= 1.0 + step_return
    max_equity = max(equity, max_equity)
    if drawdown_penalty:
        reward -= drawdown_penalty * max(0.0, (max_equity - equity) / max_equity)

    return (
        float(reward * reward_scale),
        float(step_return),
        float(equity),
        float(max_equity),
    )


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
    prep = lambda x: (
        np.asarray(x, dtype=np.float32).reshape(-1) if x is not None else None
    )
    open_, high, low = prep(open_), prep(high), prep(low)

    rewards = np.zeros(len(actions), dtype=np.float32)
    if len(actions) < 2:
        return rewards

    equity, max_equity, prev_action = 1.0, 1.0, 0.0
    for i in range(len(actions) - 1):
        o_t, h_t, l_t = (x[i] if x is not None else None for x in (open_, high, low))
        o_t1, h_t1, l_t1 = (
            x[i + 1] if x is not None else None for x in (open_, high, low)
        )

        reward, _, equity, max_equity = compute_step_reward(
            actions[i],
            prev_action,
            close[i],
            close[i + 1],
            fee,
            slip,
            reward_scale,
            turnover_penalty,
            position_penalty,
            drawdown_penalty,
            equity,
            max_equity,
            price_mode,
            o_t,
            h_t,
            l_t,
            o_t1,
            h_t1,
            l_t1,
            range_penalty,
        )
        rewards[i] = reward
        prev_action = actions[i]
    return rewards


def compute_rtg(rewards, gamma=1.0):
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    rtg = np.zeros_like(rewards)
    run = 0.0
    for i in reversed(range(len(rewards))):
        run = rewards[i] + gamma * run
        rtg[i] = run
    return rtg


def update_rtg(current_rtg, reward, gamma=1.0):
    return float((current_rtg - reward) / gamma) if gamma != 0 else 0.0

import numpy as np


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
):
    ret = close_t1 / close_t - 1.0
    delta = action - prev_action
    trade_cost = (fee + slip) * abs(delta)
    step_return = action * ret - trade_cost
    reward = step_return
    if turnover_penalty:
        reward -= turnover_penalty * abs(delta)
    if position_penalty:
        reward -= position_penalty * abs(action)

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
):
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)
    close = np.asarray(close, dtype=np.float32).reshape(-1)
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

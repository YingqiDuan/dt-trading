import numpy as np


class MarketEnv:
    def __init__(
        self,
        states,
        close,
        timestamps,
        fee,
        slip,
        episode_len=None,
        reward_scale=1.0,
        turnover_penalty=0.0,
        position_penalty=0.0,
        drawdown_penalty=0.0,
        action_mode="discrete",
        rng=None,
    ):
        self.states = np.asarray(states, dtype=np.float32)
        self.close = np.asarray(close, dtype=np.float32)
        self.timestamps = np.asarray(timestamps, dtype=np.int64)
        self.fee = float(fee)
        self.slip = float(slip)
        self.episode_len = int(episode_len) if episode_len is not None else None
        self.reward_scale = float(reward_scale)
        self.turnover_penalty = float(turnover_penalty)
        self.position_penalty = float(position_penalty)
        self.drawdown_penalty = float(drawdown_penalty)
        self.action_mode = action_mode
        self.rng = rng or np.random.RandomState(0)

        if len(self.states) < 2:
            raise ValueError("market env requires at least 2 timesteps")

        self.idx = 0
        self.end_idx = len(self.states) - 1
        self.position = 0.0
        self.equity = 1.0
        self.max_equity = 1.0

    def _sample_start(self):
        max_start = len(self.states) - 2
        if self.episode_len is not None:
            max_start = min(max_start, len(self.states) - 1 - self.episode_len)
        if max_start < 0:
            raise ValueError("episode_len too long for available data")
        return int(self.rng.randint(0, max_start + 1))

    def reset(self, start_idx=None):
        if start_idx is None:
            start_idx = self._sample_start()
        if start_idx < 0 or start_idx >= len(self.states) - 1:
            raise ValueError("start_idx out of range")

        self.idx = int(start_idx)
        if self.episode_len is None:
            self.end_idx = len(self.states) - 1
        else:
            self.end_idx = min(len(self.states) - 1, self.idx + self.episode_len)
        self.position = 0.0
        self.equity = 1.0
        self.max_equity = 1.0
        return self.states[self.idx]

    def step(self, action):
        if self.idx >= self.end_idx:
            raise RuntimeError("step called after episode done")

        if self.action_mode == "continuous":
            action = float(np.clip(action, -1.0, 1.0))
        else:
            action = int(np.clip(action, -1, 1))

        ret = self.close[self.idx + 1] / self.close[self.idx] - 1.0
        delta = action - self.position
        trade_cost = (self.fee + self.slip) * abs(delta)
        pnl = action * ret
        step_return = pnl - trade_cost
        reward = step_return
        if self.turnover_penalty:
            reward -= self.turnover_penalty * abs(delta)
        if self.position_penalty:
            reward -= self.position_penalty * abs(action)

        self.position = action
        self.equity *= 1.0 + step_return
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        if self.drawdown_penalty:
            drawdown = max(0.0, (self.max_equity - self.equity) / self.max_equity)
            reward -= self.drawdown_penalty * drawdown

        reward *= self.reward_scale

        self.idx += 1
        done = self.idx >= self.end_idx
        next_state = self.states[self.idx]
        info = {
            "timestamp": int(self.timestamps[self.idx]),
            "equity": float(self.equity),
            "position": float(self.position),
        }
        return next_state, float(reward), done, info

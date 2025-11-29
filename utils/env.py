import numpy as np
import gymnasium
from gymnasium import spaces
from scipy.stats import norm
from .gbm import simulate_gbm_path
from .bsm import bs_call_price

class HedgingEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for option hedging using RL.
    The agent learns to adjust its hedge dynamically to minimize risk/cost.
    """

    metadata = {"render_modes": ["human"]}
    def __init__(self, S0=100, K=100, mu=0.05, sigma=0.2, T=1.0, steps=252, cost=0.001, reward_type="hedge_error", hedge_step=0.3):
        super(HedgingEnv, self).__init__()

        self.S0, self.K, self.mu, self.sigma, self.T, self.steps, self.cost, self.reward_type = S0, K, mu, sigma, T, steps, cost, reward_type
        self.dt = T / steps
        self.returns_window = []
        self.hedge_step = hedge_step

        # Observation space: [normalized price, time to maturity, current hedge]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0], dtype=np.float32),
        )

        # Action space: continuous scalar (hedge adjustment)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reset(seed=None)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate new path through GBM
        self.S = simulate_gbm_path(S0=self.S0, mu=self.mu, sigma=self.sigma, dt=self.dt, steps=self.steps)
        self.t = 0
        self.hedge = 0.0
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.done = False

        return self._get_obs(), {}

    def _get_obs(self):
        normalised_price = self.S[self.t] / self.S0
        time_to_maturity = 1.0 - (self.t / self.steps)
        return np.array([normalised_price, time_to_maturity, self.hedge], dtype=np.float32)

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        prev_hedge = self.hedge
        self.hedge = np.clip(prev_hedge + action * self.hedge_step, -1.0, 1.0)

        delta_change = self.hedge - prev_hedge
        self.cash += delta_change * self.S[self.t] - abs(delta_change) * self.cost * self.S[self.t]
        self.cash *= np.exp(self.mu * self.dt)

        self.t += 1
        terminated = self.t >= self.steps - 1

        tau = self.T - self.t * self.dt
        option_value = bs_call_price(S=self.S[self.t], K=self.K, r=self.mu, sigma=self.sigma, T=tau)
        portfolio_value = option_value - self.hedge * self.S[self.t] + self.cash
        reward = self._calculate_reward(portfolio_value=portfolio_value)
        self.portfolio_value = portfolio_value

        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def _calculate_reward(self, portfolio_value):
        if self.reward_type == "pnl":
            return portfolio_value - self.portfolio_value

        elif self.reward_type == "hedge_error":
            return -(portfolio_value - self.portfolio_value)**2

        elif self.reward_type == "sharpe":
            self.returns_window.append(portfolio_value - self.portfolio_value)
            if len(self.returns_window) > 20:
                vol = np.std(self.returns_window[-20:])
                return (portfolio_value - self.portfolio_value) / (vol + 1e-6)
            else:
                return portfolio_value - self.portfolio_value
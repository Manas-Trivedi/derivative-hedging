import gymnasium
import numpy as np
from gymnasium import spaces

from .bsm import bs_call_price, bs_delta
from .gbm import simulate_gbm_path


def build_observation(price, S0, K, mu, sigma, tau, hedge, total_T=1.0):
    safe_price = max(float(price), 1e-12)
    safe_strike = max(float(K), 1e-12)
    safe_tau = max(float(tau), 0.0)
    option_value = bs_call_price(S=safe_price, K=safe_strike, r=mu, sigma=sigma, T=safe_tau)
    target_delta = bs_delta(safe_price, safe_strike, mu, sigma, max(safe_tau, 1e-12))
    log_moneyness = np.log(safe_price / safe_strike)

    return np.array(
        [
            safe_price / S0,
            safe_tau / max(float(total_T), 1e-12),
            hedge,
            log_moneyness,
            option_value / S0,
            target_delta,
        ],
        dtype=np.float32,
    )


class HedgingEnv(gymnasium.Env):
    """
    Custom Gymnasium environment for option hedging using RL.
    The agent learns to adjust its hedge dynamically to minimize risk/cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        S0=100,
        K=100,
        mu=0.05,
        sigma=0.2,
        T=1.0,
        steps=252,
        cost=0.001,
        reward_type="combined",
        hedge_step=0.1,
        action_mode="target",
        delta_reward_scale=1.0,
        tc_penalty_scale=1.0,
    ):
        super(HedgingEnv, self).__init__()

        self.S0 = S0
        self.K = K
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.cost = cost
        self.reward_type = reward_type
        self.dt = T / steps
        self.hedge_step = hedge_step
        self.action_mode = action_mode
        self.delta_reward_scale = delta_reward_scale
        self.tc_penalty_scale = tc_penalty_scale

        # Observation space: [normalized price, time to maturity, current hedge,
        # log-moneyness, normalized option value, Black-Scholes delta].
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -5.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 1.0, 1.0, 5.0, 5.0, 1.0], dtype=np.float32),
        )

        # Action space: either direct hedge target or hedge adjustment.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reset(seed=None)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.S = simulate_gbm_path(
            S0=self.S0,
            mu=self.mu,
            sigma=self.sigma,
            dt=self.dt,
            steps=self.steps,
        )
        self.t = 0
        self.hedge = 0.0
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.returns_window = []

        return self._get_obs(), {}

    def _time_to_maturity(self):
        return max(self.T - self.t * self.dt, 0.0)

    def _target_delta(self):
        return float(
            bs_delta(
                self.S[self.t],
                self.K,
                self.mu,
                self.sigma,
                max(self._time_to_maturity(), 1e-12),
            )
        )

    def _portfolio_value(self):
        option_value = bs_call_price(
            S=self.S[self.t],
            K=self.K,
            r=self.mu,
            sigma=self.sigma,
            T=self._time_to_maturity(),
        )
        return option_value - self.hedge * self.S[self.t] + self.cash

    def _get_obs(self):
        return build_observation(
            price=self.S[self.t],
            S0=self.S0,
            K=self.K,
            mu=self.mu,
            sigma=self.sigma,
            tau=self._time_to_maturity(),
            hedge=self.hedge,
            total_T=self.T,
        )

    def step(self, action):
        action_value = float(np.clip(action[0], -1.0, 1.0))
        prev_hedge = self.hedge

        if self.action_mode == "target":
            self.hedge = action_value
        elif self.action_mode == "adjust":
            self.hedge = float(np.clip(prev_hedge + action_value * self.hedge_step, -1.0, 1.0))
        else:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        delta_change = self.hedge - prev_hedge
        trading_cost = abs(delta_change) * self.cost * self.S[self.t]
        self.cash += delta_change * self.S[self.t] - trading_cost
        self.cash *= np.exp(self.mu * self.dt)

        self.t += 1
        terminated = self.t >= self.steps - 1

        portfolio_value = self._portfolio_value()
        reward = self._calculate_reward(portfolio_value=portfolio_value, trading_cost=trading_cost)
        self.portfolio_value = portfolio_value

        info = {
            "portfolio_value": portfolio_value,
            "hedge": self.hedge,
            "target_delta": self._target_delta(),
            "trading_cost": trading_cost,
        }
        return self._get_obs(), reward, terminated, False, info

    def _calculate_reward(self, portfolio_value, trading_cost):
        step_pnl = portfolio_value - self.portfolio_value
        delta_error = self.hedge - self._target_delta()

        if self.reward_type == "pnl":
            return step_pnl - self.tc_penalty_scale * trading_cost

        if self.reward_type == "hedge_error":
            return -(step_pnl ** 2) - self.tc_penalty_scale * trading_cost

        if self.reward_type == "delta_tracking":
            return -(delta_error ** 2) - self.tc_penalty_scale * trading_cost

        if self.reward_type == "combined":
            return (
                -(step_pnl ** 2)
                - self.delta_reward_scale * (delta_error ** 2)
                - self.tc_penalty_scale * trading_cost
            )

        if self.reward_type == "sharpe":
            self.returns_window.append(step_pnl)
            if len(self.returns_window) > 20:
                vol = np.std(self.returns_window[-20:])
                return (step_pnl / (vol + 1e-6)) - self.tc_penalty_scale * trading_cost
            return step_pnl - self.tc_penalty_scale * trading_cost

        raise ValueError(f"Unsupported reward_type: {self.reward_type}")

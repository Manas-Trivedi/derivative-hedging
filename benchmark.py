import contextlib
import io
import json
import warnings
from pathlib import Path

# Suppress noisy third-party import output to keep CLI results readable.
_stderr_buf = io.StringIO()
with contextlib.redirect_stderr(_stderr_buf):
    import numpy as np
    import typer
    from stable_baselines3 import PPO

from utils.bsm import bs_call_price, bs_delta
from utils.env import build_observation

warnings.filterwarnings("ignore", category=UserWarning)


MetricDict = dict[str, float]
StrategySummary = dict[str, MetricDict]


def simulate_gbm_path_with_rng(
    rng: np.random.Generator,
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    steps: int,
) -> np.ndarray:
    prices = np.zeros(steps, dtype=float)
    prices[0] = S0
    for t in range(1, steps):
        z = rng.normal()
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return prices


def rollout_ppo(
    model: PPO,
    prices: np.ndarray,
    S0: float,
    K: float,
    mu: float,
    sigma: float,
    T: float,
    cost: float,
    hedge_step: float,
    action_mode: str,
) -> dict[str, np.ndarray]:
    steps = len(prices)
    dt = T / steps
    hedge = 0.0
    cash = 0.0
    portfolio_values = []
    hedge_positions = []
    trade_sizes = []
    transaction_costs = []
    target_deltas = []

    for t in range(steps - 1):
        tau = max(T - t * dt, 0.0)
        obs = build_observation(
            price=prices[t],
            S0=S0,
            K=K,
            mu=mu,
            sigma=sigma,
            tau=tau,
            hedge=hedge,
            total_T=T,
        )
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(np.asarray(action).reshape(-1)[0])
        action_value = float(np.clip(action_value, -1.0, 1.0))

        prev_hedge = hedge
        if action_mode == "target":
            hedge = action_value
        else:
            hedge = float(np.clip(prev_hedge + action_value * hedge_step, -1.0, 1.0))

        delta_change = hedge - prev_hedge
        trading_cost = abs(delta_change) * cost * prices[t]

        cash += delta_change * prices[t] - trading_cost
        cash *= np.exp(mu * dt)

        next_tau = max(T - (t + 1) * dt, 0.0)
        option_value = bs_call_price(S=prices[t + 1], K=K, r=mu, sigma=sigma, T=next_tau)
        target_delta = float(bs_delta(prices[t + 1], K, mu, sigma, max(next_tau, 1e-12)))
        portfolio_value = option_value - hedge * prices[t + 1] + cash

        hedge_positions.append(hedge)
        trade_sizes.append(abs(delta_change))
        transaction_costs.append(trading_cost)
        portfolio_values.append(portfolio_value)
        target_deltas.append(target_delta)

    return {
        "portfolio_values": np.asarray(portfolio_values, dtype=float),
        "hedge_positions": np.asarray(hedge_positions, dtype=float),
        "trade_sizes": np.asarray(trade_sizes, dtype=float),
        "transaction_costs": np.asarray(transaction_costs, dtype=float),
        "target_deltas": np.asarray(target_deltas, dtype=float),
    }


def rollout_delta_hedge(
    prices: np.ndarray,
    K: float,
    mu: float,
    sigma: float,
    T: float,
    cost: float,
) -> dict[str, np.ndarray]:
    steps = len(prices)
    dt = T / steps
    hedge = 0.0
    cash = 0.0
    portfolio_values = []
    hedge_positions = []
    trade_sizes = []
    transaction_costs = []
    target_deltas = []

    for t in range(steps - 1):
        tau = max(T - t * dt, 1e-12)
        target_hedge = float(np.clip(bs_delta(prices[t], K, mu, sigma, tau), -1.0, 1.0))

        prev_hedge = hedge
        hedge = target_hedge
        delta_change = hedge - prev_hedge
        trading_cost = abs(delta_change) * cost * prices[t]

        cash += delta_change * prices[t] - trading_cost
        cash *= np.exp(mu * dt)

        next_tau = max(T - (t + 1) * dt, 0.0)
        option_value = bs_call_price(S=prices[t + 1], K=K, r=mu, sigma=sigma, T=next_tau)
        next_target_delta = float(bs_delta(prices[t + 1], K, mu, sigma, max(next_tau, 1e-12)))
        portfolio_value = option_value - hedge * prices[t + 1] + cash

        hedge_positions.append(hedge)
        trade_sizes.append(abs(delta_change))
        transaction_costs.append(trading_cost)
        portfolio_values.append(portfolio_value)
        target_deltas.append(next_target_delta)

    return {
        "portfolio_values": np.asarray(portfolio_values, dtype=float),
        "hedge_positions": np.asarray(hedge_positions, dtype=float),
        "trade_sizes": np.asarray(trade_sizes, dtype=float),
        "transaction_costs": np.asarray(transaction_costs, dtype=float),
        "target_deltas": np.asarray(target_deltas, dtype=float),
    }


def compute_metrics(rollout: dict[str, np.ndarray]) -> MetricDict:
    portfolio_values = np.asarray(rollout["portfolio_values"], dtype=float)
    trade_sizes = np.asarray(rollout["trade_sizes"], dtype=float)
    transaction_costs = np.asarray(rollout["transaction_costs"], dtype=float)
    hedge_positions = np.asarray(rollout["hedge_positions"], dtype=float)
    target_deltas = np.asarray(rollout["target_deltas"], dtype=float)
    hedge_gap = hedge_positions - target_deltas

    if len(portfolio_values) <= 1:
        step_pnl = np.zeros(1, dtype=float)
        terminal_pnl = float(portfolio_values[-1]) if len(portfolio_values) else 0.0
    else:
        step_pnl = np.diff(portfolio_values)
        terminal_pnl = float(portfolio_values[-1] - portfolio_values[0])

    return {
        "hedging_mse": float(np.mean(step_pnl**2)),
        "pnl_variance": float(np.var(step_pnl)),
        "mean_pnl_drift": float(np.mean(step_pnl)),
        "mean_abs_step_pnl": float(np.mean(np.abs(step_pnl))),
        "terminal_pnl": terminal_pnl,
        "terminal_abs_pnl": abs(terminal_pnl),
        "hedge_gap_mse": float(np.mean(hedge_gap**2)) if hedge_gap.size else 0.0,
        "mean_abs_hedge_gap": float(np.mean(np.abs(hedge_gap))) if hedge_gap.size else 0.0,
        "avg_turnover": float(np.mean(trade_sizes)) if trade_sizes.size else 0.0,
        "avg_transaction_cost": float(np.mean(transaction_costs)) if transaction_costs.size else 0.0,
    }


def summarise_metrics(metrics_by_strategy: dict[str, list[MetricDict]]) -> StrategySummary:
    summary: StrategySummary = {}
    for strategy, rows in metrics_by_strategy.items():
        metric_names = rows[0].keys()
        summary[strategy] = {
            name: float(np.mean([row[name] for row in rows]))
            for name in metric_names
        }
    return summary


def improvement_vs_baseline(ppo_value: float, baseline_value: float, lower_is_better: bool = True) -> float:
    if np.isclose(baseline_value, 0.0):
        return float("nan")
    if lower_is_better:
        return 100.0 * (baseline_value - ppo_value) / abs(baseline_value)
    return 100.0 * (ppo_value - baseline_value) / abs(baseline_value)


def print_summary(summary: StrategySummary) -> None:
    metrics = [
        ("hedging_mse", "lower"),
        ("pnl_variance", "lower"),
        ("mean_pnl_drift", "closer to 0"),
        ("mean_abs_step_pnl", "lower"),
        ("terminal_pnl", "closer to 0"),
        ("terminal_abs_pnl", "lower"),
        ("hedge_gap_mse", "lower"),
        ("mean_abs_hedge_gap", "lower"),
        ("avg_turnover", "lower"),
        ("avg_transaction_cost", "lower"),
    ]

    print("\nBenchmark Results")
    print("-" * 86)
    print(f"{'Metric':<24} {'PPO':>16} {'Delta Hedge':>16} {'Winner':>14} {'PPO vs Delta':>14}")
    print("-" * 86)

    for metric, preference in metrics:
        ppo_value = summary["ppo"][metric]
        delta_value = summary["delta"][metric]

        if preference == "closer to 0":
            ppo_score = abs(ppo_value)
            delta_score = abs(delta_value)
            lower_is_better = True
        else:
            ppo_score = ppo_value
            delta_score = delta_value
            lower_is_better = True

        winner = "PPO" if ppo_score < delta_score else "Delta"
        delta_pct = improvement_vs_baseline(ppo_score, delta_score, lower_is_better=lower_is_better)
        delta_pct_display = "n/a" if np.isnan(delta_pct) else f"{delta_pct:+.2f}%"

        print(
            f"{metric:<24} "
            f"{ppo_value:>16.6f} "
            f"{delta_value:>16.6f} "
            f"{winner:>14} "
            f"{delta_pct_display:>14}"
        )

    print("-" * 86)
    print("Primary win criteria: lower hedging MSE and lower PnL variance.")
    print("Hedge-gap metrics show how closely PPO tracks the Black-Scholes teacher signal.")


def benchmark(
    model_path: str = typer.Option("models/ppo_hedging_best.zip", help="Path to the trained PPO model."),
    episodes: int = typer.Option(250, help="Number of shared GBM paths to evaluate."),
    S0: float = typer.Option(100.0, help="Initial stock price."),
    K: float = typer.Option(100.0, help="Option strike."),
    mu: float = typer.Option(0.05, help="Drift / rate used by the existing environment."),
    sigma: float = typer.Option(0.2, help="Underlying volatility."),
    T: float = typer.Option(1.0, help="Time to maturity in years."),
    steps: int = typer.Option(252, help="Number of simulation steps per path."),
    cost: float = typer.Option(0.001, help="Transaction-cost rate per unit traded."),
    hedge_step: float = typer.Option(0.1, help="Per-step hedge adjustment size used when action_mode=adjust."),
    action_mode: str = typer.Option("target", help="Action semantics expected by the PPO model: target or adjust."),
    seed: int = typer.Option(42, help="Random seed for reproducible shared paths."),
    json_out: Path | None = typer.Option(None, help="Optional path to write the benchmark summary as JSON."),
) -> None:
    """Benchmark PPO against Black-Scholes delta hedging on identical simulated paths."""
    if episodes <= 0:
        raise typer.BadParameter("episodes must be positive")
    if steps < 3:
        raise typer.BadParameter("steps must be at least 3")
    if action_mode not in {"target", "adjust"}:
        raise typer.BadParameter("action_mode must be either 'target' or 'adjust'")

    model = PPO.load(model_path)
    rng = np.random.default_rng(seed)
    dt = T / steps

    metrics_by_strategy: dict[str, list[MetricDict]] = {"ppo": [], "delta": []}

    for _ in range(episodes):
        prices = simulate_gbm_path_with_rng(rng=rng, S0=S0, mu=mu, sigma=sigma, dt=dt, steps=steps)

        ppo_rollout = rollout_ppo(
            model=model,
            prices=prices,
            S0=S0,
            K=K,
            mu=mu,
            sigma=sigma,
            T=T,
            cost=cost,
            hedge_step=hedge_step,
            action_mode=action_mode,
        )
        delta_rollout = rollout_delta_hedge(
            prices=prices,
            K=K,
            mu=mu,
            sigma=sigma,
            T=T,
            cost=cost,
        )

        metrics_by_strategy["ppo"].append(compute_metrics(ppo_rollout))
        metrics_by_strategy["delta"].append(compute_metrics(delta_rollout))

    summary = summarise_metrics(metrics_by_strategy)
    print_summary(summary)

    if json_out is not None:
        json_out.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved JSON summary to {json_out}")


if __name__ == "__main__":
    typer.run(benchmark)

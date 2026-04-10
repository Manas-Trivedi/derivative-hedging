import contextlib
import io
import json
import warnings
from pathlib import Path

# Capture stderr during imports to suppress packages that print deprecation messages
_stderr_buf = io.StringIO()
with contextlib.redirect_stderr(_stderr_buf):
    import matplotlib.pyplot as plt
    import numpy as np
    import typer
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from utils.bsm import bs_delta
    from utils.env import HedgingEnv

# Also ignore related UserWarnings (if any libraries use warnings.warn)
warnings.filterwarnings("ignore", category=UserWarning)

app = typer.Typer(help="CLI for Reinforcement Learning-based Derivative Hedging")


def parse_net_arch(net_arch: str) -> list[int]:
    return [int(width.strip()) for width in net_arch.split(",") if width.strip()]


def build_env_kwargs(
    reward_type: str,
    action_mode: str,
    hedge_step: float,
    delta_reward_scale: float,
    tc_penalty_scale: float,
    S0: float,
    K: float,
    mu: float,
    sigma: float,
    T: float,
    steps_per_episode: int,
    cost: float,
) -> dict:
    return {
        "reward_type": reward_type,
        "action_mode": action_mode,
        "hedge_step": hedge_step,
        "delta_reward_scale": delta_reward_scale,
        "tc_penalty_scale": tc_penalty_scale,
        "S0": S0,
        "K": K,
        "mu": mu,
        "sigma": sigma,
        "T": T,
        "steps": steps_per_episode,
        "cost": cost,
    }


def make_env(env_kwargs: dict, seed: int | None = None):
    def _factory():
        env = HedgingEnv(**env_kwargs)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)

    return _factory


def evaluate_agent(model, env_kwargs, n_episodes=50):
    rewards = []
    finals = []

    for episode in range(n_episodes):
        env = HedgingEnv(**env_kwargs)
        obs, _ = env.reset(seed=episode)
        done, truncated = False, False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward  # type: ignore

        rewards.append(total_reward)
        finals.append(env.portfolio_value)

    return {"mean_reward": float(np.mean(rewards)), "mean_final_value": float(np.mean(finals))}


def diagnose_agent(model, env_kwargs, n_episodes=50):
    episode_rows = []

    for episode in range(n_episodes):
        env = HedgingEnv(**env_kwargs)
        obs, _ = env.reset(seed=episode)
        done, truncated = False, False
        total_reward = 0.0
        hedge_gaps = []
        trade_sizes = []
        transaction_costs = []

        while not (done or truncated):
            prev_hedge = env.hedge
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward  # type: ignore
            hedge_gaps.append(info["hedge"] - info["target_delta"])
            trade_sizes.append(abs(info["hedge"] - prev_hedge))
            transaction_costs.append(info["trading_cost"])

        episode_rows.append(
            {
                "reward": total_reward,
                "final_value": env.portfolio_value,
                "hedge_gap_mse": float(np.mean(np.square(hedge_gaps))) if hedge_gaps else 0.0,
                "mean_abs_hedge_gap": float(np.mean(np.abs(hedge_gaps))) if hedge_gaps else 0.0,
                "terminal_abs_hedge_gap": float(abs(hedge_gaps[-1])) if hedge_gaps else 0.0,
                "avg_turnover": float(np.mean(trade_sizes)) if trade_sizes else 0.0,
                "avg_transaction_cost": float(np.mean(transaction_costs)) if transaction_costs else 0.0,
            }
        )

    metric_names = episode_rows[0].keys()
    return {
        metric: float(np.mean([row[metric] for row in episode_rows]))
        for metric in metric_names
    }


def compare_hedge(model_path="models/ppo_hedging_best", steps=252, env_kwargs=None):
    model = PPO.load(model_path)
    env_kwargs = env_kwargs or {}
    env = HedgingEnv(steps=steps, **env_kwargs)
    obs, _ = env.reset()
    done, truncated = False, False

    rl_deltas, bsm_deltas = [], []
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        tau = env.T - env.t * env.dt
        rl_deltas.append(env.hedge)
        bsm_deltas.append(bs_delta(env.S[env.t], env.K, env.mu, env.sigma, max(tau, 1e-12)))

    plt.plot(rl_deltas, label="RL hedge")
    plt.plot(bsm_deltas, label="BSM delta")
    plt.legend()
    plt.title("Hedging Comparison - RL vs Analytic")
    plt.show()


@app.command()
def train(
    steps: int = typer.Option(1_000_000, help="Number of training timesteps"),
    reward_type: str = typer.Option("combined", help="Reward mode: combined, hedge_error, delta_tracking, pnl, or sharpe"),
    model_name: str = typer.Option("ppo_hedging_model", help="Output model name"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    n_envs: int = typer.Option(8, help="Number of parallel environments"),
    n_steps: int = typer.Option(2048, help="PPO rollout length per environment"),
    batch_size: int = typer.Option(256, help="PPO minibatch size"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    gae_lambda: float = typer.Option(0.95, help="GAE lambda"),
    ent_coef: float = typer.Option(0.0, help="Entropy coefficient"),
    clip_range: float = typer.Option(0.2, help="PPO clip range"),
    net_arch: str = typer.Option("128,128", help="Comma-separated hidden layer widths"),
    eval_freq: int = typer.Option(50_000, help="Evaluation frequency in timesteps"),
    eval_episodes: int = typer.Option(25, help="Number of episodes per evaluation pass"),
    checkpoint_freq: int = typer.Option(100_000, help="Checkpoint frequency in timesteps"),
    seed: int = typer.Option(42, help="Training seed"),
    action_mode: str = typer.Option("target", help="Action semantics: target or adjust"),
    hedge_step: float = typer.Option(0.1, help="Per-step hedge adjustment size when action_mode=adjust"),
    delta_reward_scale: float = typer.Option(1.0, help="Weight on delta-tracking penalty in combined reward"),
    tc_penalty_scale: float = typer.Option(1.0, help="Weight on transaction-cost penalty"),
    S0: float = typer.Option(100.0, help="Initial stock price"),
    K: float = typer.Option(100.0, help="Option strike"),
    mu: float = typer.Option(0.05, help="Drift / rate used by the environment"),
    sigma: float = typer.Option(0.2, help="Underlying volatility"),
    T: float = typer.Option(1.0, help="Time to maturity in years"),
    steps_per_episode: int = typer.Option(252, help="Number of simulation steps per episode"),
    cost: float = typer.Option(0.001, help="Transaction cost rate per unit traded"),
):
    """Train a PPO model on the HedgingEnv."""
    env_kwargs = build_env_kwargs(
        reward_type=reward_type,
        action_mode=action_mode,
        hedge_step=hedge_step,
        delta_reward_scale=delta_reward_scale,
        tc_penalty_scale=tc_penalty_scale,
        S0=S0,
        K=K,
        mu=mu,
        sigma=sigma,
        T=T,
        steps_per_episode=steps_per_episode,
        cost=cost,
    )
    run_dir = Path("models") / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = run_dir / "best_model"
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    best_model_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)

    env = DummyVecEnv([make_env(env_kwargs, seed=seed + idx) for idx in range(n_envs)])
    eval_env = DummyVecEnv([make_env(env_kwargs, seed=seed + 10_000)])

    callbacks = CallbackList(
        [
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(run_dir),
                eval_freq=max(eval_freq // max(n_envs, 1), 1),
                n_eval_episodes=eval_episodes,
                deterministic=True,
            ),
            CheckpointCallback(
                save_freq=max(checkpoint_freq // max(n_envs, 1), 1),
                save_path=str(checkpoint_dir),
                name_prefix=model_name,
            ),
        ]
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        policy_kwargs={"net_arch": parse_net_arch(net_arch)},
        tensorboard_log=str(tensorboard_dir),
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=steps, callback=callbacks)
    final_model_path = Path("models") / f"{model_name}.zip"
    model.save(final_model_path)

    config_path = run_dir / "training_config.json"
    config_path.write_text(
        json.dumps(
            {
                "steps": steps,
                "reward_type": reward_type,
                "learning_rate": lr,
                "n_envs": n_envs,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "ent_coef": ent_coef,
                "clip_range": clip_range,
                "net_arch": parse_net_arch(net_arch),
                "eval_freq": eval_freq,
                "eval_episodes": eval_episodes,
                "checkpoint_freq": checkpoint_freq,
                "seed": seed,
                "env_kwargs": env_kwargs,
            },
            indent=2,
        )
    )

    typer.echo(f"Model trained and saved as {final_model_path}")
    typer.echo(f"Best checkpoints and logs saved under {run_dir}")


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to saved PPO model"),
    episodes: int = typer.Option(50, "-e", "--episodes", help="Number of test episodes"),
    reward: str = typer.Option("combined", "-r", "--reward", help="Reward function used for evaluation"),
    action_mode: str = typer.Option("target", help="Action semantics: target or adjust"),
    hedge_step: float = typer.Option(0.1, help="Per-step hedge adjustment size when action_mode=adjust"),
    delta_reward_scale: float = typer.Option(1.0, help="Weight on delta-tracking penalty in combined reward"),
    tc_penalty_scale: float = typer.Option(1.0, help="Weight on transaction-cost penalty"),
):
    """Evaluate a trained model."""
    model = PPO.load(model_path)
    env_kwargs = build_env_kwargs(
        reward_type=reward,
        action_mode=action_mode,
        hedge_step=hedge_step,
        delta_reward_scale=delta_reward_scale,
        tc_penalty_scale=tc_penalty_scale,
        S0=100.0,
        K=100.0,
        mu=0.05,
        sigma=0.2,
        T=1.0,
        steps_per_episode=252,
        cost=0.001,
    )
    stats = evaluate_agent(model, env_kwargs=env_kwargs, n_episodes=episodes)
    typer.echo(f"Mean reward: {stats['mean_reward']:.6f}")
    typer.echo(f"Mean final portfolio value: {stats['mean_final_value']:.6f}")


@app.command()
def diagnose(
    model_path: str = typer.Argument(..., help="Path to saved PPO model"),
    episodes: int = typer.Option(50, "-e", "--episodes", help="Number of diagnostic episodes"),
    reward: str = typer.Option("combined", "-r", "--reward", help="Reward function used for rollout"),
    action_mode: str = typer.Option("target", help="Action semantics: target or adjust"),
    hedge_step: float = typer.Option(0.1, help="Per-step hedge adjustment size when action_mode=adjust"),
    delta_reward_scale: float = typer.Option(1.0, help="Weight on delta-tracking penalty in combined reward"),
    tc_penalty_scale: float = typer.Option(1.0, help="Weight on transaction-cost penalty"),
    json_out: Path | None = typer.Option(None, help="Optional path to write the diagnostic summary as JSON"),
):
    """Run hedge-tracking diagnostics on a trained model."""
    model = PPO.load(model_path)
    env_kwargs = build_env_kwargs(
        reward_type=reward,
        action_mode=action_mode,
        hedge_step=hedge_step,
        delta_reward_scale=delta_reward_scale,
        tc_penalty_scale=tc_penalty_scale,
        S0=100.0,
        K=100.0,
        mu=0.05,
        sigma=0.2,
        T=1.0,
        steps_per_episode=252,
        cost=0.001,
    )
    summary = diagnose_agent(model, env_kwargs=env_kwargs, n_episodes=episodes)

    typer.echo("Diagnostic Summary")
    typer.echo("-" * 72)
    for metric, value in summary.items():
        typer.echo(f"{metric:<24} {value:>16.6f}")

    if json_out is not None:
        json_out.write_text(json.dumps(summary, indent=2))
        typer.echo(f"\nSaved diagnostic summary to {json_out}")


@app.command()
def compare(
    model_path: str = typer.Argument(..., help="Path to PPO model"),
    steps: int = typer.Option(252, help="Number of time steps to simulate"),
    action_mode: str = typer.Option("target", help="Action semantics: target or adjust"),
    hedge_step: float = typer.Option(0.1, help="Per-step hedge adjustment size when action_mode=adjust"),
    reward: str = typer.Option("combined", help="Reward function used for rollout"),
):
    """Compare RL hedge vs analytic Black-Scholes hedge visually."""
    compare_hedge(
        model_path,
        steps=steps,
        env_kwargs={
            "action_mode": action_mode,
            "hedge_step": hedge_step,
            "reward_type": reward,
        },
    )


if __name__ == "__main__":
    app()

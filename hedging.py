import contextlib
import io
import sys
import warnings

# Capture stderr during imports to suppress packages that print deprecation messages
_stderr_buf = io.StringIO()
with contextlib.redirect_stderr(_stderr_buf):
    import typer
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.env import HedgingEnv
    from utils.bsm import bs_delta

# Also ignore related UserWarnings (if any libraries use warnings.warn)
warnings.filterwarnings("ignore", category=UserWarning)

app = typer.Typer(help="CLI for Reinforcement Learning-based Derivative Hedging")

def evaluate_agent(model, n_episodes=50, reward_type="hedge_error"):
    rewards, finals = [], []
    for _ in range(n_episodes):
        env = HedgingEnv(reward_type=reward_type)
        obs, info = env.reset()
        done, truncated = False, False
        total_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward # type: ignore
        rewards.append(total_reward)
        finals.append(env.portfolio_value)
    return {"mean_reward": np.mean(rewards), "mean_final_value": np.mean(finals)}

def compare_hedge(model_path = "models/ppo_hedging_best", steps=252):
    model = PPO.load(model_path)
    env = HedgingEnv(steps=steps)
    obs, info = env.reset()
    done, truncated = False, False

    rl_deltas, bsm_deltas, prices = [], [], []
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        tau = env.T - env.t * env.dt
        prices.append(env.S[env.t])
        rl_deltas.append(env.hedge)
        bsm_deltas.append(bs_delta(env.S[env.t], env.K, env.mu, env.sigma, tau))

    plt.plot(rl_deltas, label="RL Δ (Agent)")
    plt.plot(bsm_deltas, label="BSM Δ (Analytic)")
    plt.legend()
    plt.title("Hedging Comparison — RL vs Analytic")
    plt.show()

@app.command()
def train(
    steps: int = typer.Option(200_000, help="Number of training timesteps"),
    reward_type: str = typer.Option("hedge_error", help="Reward mode: hedge_error, pnl, or sharpe"),
    model_name: str = typer.Option("ppo_hedging_model", help="Output model name"),
    lr: float = typer.Option(3e-4, help="Learning rate")
):
    """Train a PPO model on the HedgingEnv."""
    env = DummyVecEnv([lambda: Monitor(HedgingEnv(reward_type=reward_type))])
    model = PPO("MlpPolicy", env, learning_rate=lr, verbose=1)
    model.learn(total_timesteps=steps)
    model.save(f"models/{model_name}.zip")
    typer.echo(f"Model trained and saved as models/{model_name}.zip")

@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to saved PPO model"),
    episodes: int = typer.Option(50, "-e", "--episodes", help="Number of test episodes"),
    reward: str = typer.Option("hedge_error", "-r", "--reward", help="Reward function to be used for evaluation")
):
    """Evaluate a trained model."""
    model = PPO.load(model_path)
    stats = evaluate_agent(model, n_episodes=episodes, reward_type=reward)
    typer.echo(f"Mean reward: {stats['mean_reward']:.3f}")
    typer.echo(f"Mean final portfolio value: {stats['mean_final_value']:.3f}")

@app.command()
def compare(
    model_path: str = typer.Argument(..., help="Path to PPO model"),
    steps: int = typer.Option(252, help="Number of time steps to simulate")
):
    """Compare RL hedge vs analytic Black-Scholes hedge visually."""
    compare_hedge(model_path, steps=steps)

if __name__ == "__main__":
    app()

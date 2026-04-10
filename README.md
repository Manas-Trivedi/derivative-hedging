# Derivative Hedging

Reinforcement learning for option pricing and portfolio hedging using Geometric Brownian Motion (GBM) and the Black-Scholes model.

## Project Overview

This project covers:
- Stock price simulation via GBM
- European option pricing (Black-Scholes)
- Delta hedging strategies
- RL-based dynamic hedging (PPO agent)
- CLI for training, evaluation, and comparison
- Diagnostic tooling for hedge-gap analysis and benchmark tracking

## Project Structure

```
hedging.py                # CLI for training, evaluation, and comparison
benchmark.py              # PPO vs delta-hedging benchmark script
requirements.txt          # Python dependencies
models/                   # Saved RL models
notebooks/                # Exploratory and training notebooks
	01_gbm_simulation.ipynb          # GBM price path simulations
	02_black_scholes_pricing.ipynb   # Option pricing & Greeks
	03_portfolio_and_hedging.ipynb   # Hedging strategies (long call + short stock)
	04_RL_Hedging_Env.ipynb          # RL environment for hedging
	05_RL_Training_and_Compare.ipynb # PPO training and comparison
	06_Eval_and_Tuning.ipynb         # Hyperparameter tuning & evaluation
utils/
	gbm.py                # GBM simulation utilities
	bsm.py                # Black-Scholes pricing & Greeks
	env.py                # RL environment class
```

## Quick Start

Set up the environment:
```zsh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## CLI Usage

Run the CLI for training, evaluation, and comparison:

```zsh
python hedging.py train --steps 1000000 --reward-type combined --model-name ppo_hedging_v2
python hedging.py evaluate models/ppo_hedging_v2.zip --episodes 50 --reward combined
python hedging.py diagnose models/ppo_hedging_v2.zip --episodes 50 --json-out diagnostics.json
python hedging.py compare models/ppo_hedging_v2.zip --steps 252
```

Recommended retraining command:

```zsh
python hedging.py train \
  --steps 1000000 \
  --model-name ppo_hedging_v2 \
  --reward-type combined \
  --action-mode target \
  --n-envs 8 \
  --n-steps 2048 \
  --batch-size 256 \
  --net-arch 128,128 \
  --eval-freq 50000 \
  --checkpoint-freq 100000
```

Training now saves:
- `models/<model_name>.zip`: final PPO model
- `models/<model_name>/best_model/best_model.zip`: best checkpoint picked by evaluation callback
- `models/<model_name>/checkpoints/`: intermediate checkpoints
- `models/<model_name>/training_config.json`: reproducible training settings
- `models/<model_name>/tensorboard/`: TensorBoard logs

Run the benchmark to compare PPO against Black-Scholes delta hedging on the same simulated paths:

```zsh
# CLI Output
python benchmark.py --model-path models/ppo_hedging_v2.zip --episodes 250 --action-mode target
# JSON Export
python benchmark.py --model-path models/ppo_hedging_v2.zip --episodes 250 --action-mode target --json-out benchmark_results.json
```

The benchmark reports:
- Mean squared hedging error (`hedging_mse`)
- Variance of step PnL (`pnl_variance`)
- Mean PnL drift
- Terminal PnL / absolute terminal PnL
- Hedge-gap tracking metrics versus analytic Black-Scholes delta
- Turnover and average transaction cost

The PPO observation space now includes:
- Normalized price
- Time to maturity
- Current hedge
- Log-moneyness
- Normalized option value
- Black-Scholes delta teacher signal

Notes before retraining:
- Existing saved PPO models were trained on the old 3-feature observation space and are not compatible with the new policy input shape.
- The new default setup trains PPO to predict the hedge target directly (`--action-mode target`) and uses a combined reward that penalizes both hedge slippage and transaction cost.

## Notebooks

- **01_gbm_simulation.ipynb**: Simulate stock price paths using GBM
- **02_black_scholes_pricing.ipynb**: Calculate European option prices and delta values
- **03_portfolio_and_hedging.ipynb**: Visualize hedging a long call position by shorting stock
- **04_RL_Hedging_Env.ipynb**: RL environment for dynamic hedging
- **05_RL_Training_and_Compare.ipynb**: Train PPO agent and compare RL hedge vs analytic hedge
- **06_Eval_and_Tuning.ipynb**: Hyperparameter tuning and final evaluation

## Features

- Modular utilities for GBM simulation and Black-Scholes pricing
- Custom RL environment for option hedging (OpenAI Gym compatible)
- PPO agent training and evaluation
- CLI for reproducible experiments
- Extensive notebook documentation and visualization

## Status

**Project complete.**
All major features implemented and tested. See notebooks and CLI for usage examples.

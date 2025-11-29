# Derivative Hedging

Reinforcement learning for option pricing and portfolio hedging using Geometric Brownian Motion (GBM) and the Black-Scholes model.

## Project Overview

This project covers:
- Stock price simulation via GBM
- European option pricing (Black-Scholes)
- Delta hedging strategies
- RL-based dynamic hedging (PPO agent)
- CLI for training, evaluation, and comparison

## Project Structure

```
hedging.py                # CLI for training, evaluation, and comparison
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
python hedging.py train --steps 200000 --reward_type hedge_error --model_name ppo_hedging_model
python hedging.py evaluate models/ppo_hedging_model.zip --episodes 50 --reward hedge_error
python hedging.py compare models/ppo_hedging_model.zip --steps 252
```

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

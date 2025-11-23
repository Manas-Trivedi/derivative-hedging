# Derivative Hedging

A reinforcement learning project exploring option pricing and portfolio hedging strategies using Geometric Brownian Motion (GBM) and the Black-Scholes model.

## Overview

This project implements simulations and pricing models to understand derivative hedging in practice, covering:
- Stock price simulations via GBM
- European option pricing (Black-Scholes)
- Delta hedging strategies on portfolios

## Project Structure

```
notebooks/
├── 01_gbm_simulation.ipynb          # GBM price path simulations
├── 02_black_scholes_pricing.ipynb   # Option pricing & Greeks
└── 03_portfolio_and_hedging.ipynb   # Hedging strategies (long call + short stock)

utils/
├── gbm.py                            # GBM simulation utilities
├── bsm.py                            # Black-Scholes pricing & Greeks
└── __init__.py
```

## Quick Start

Set up the environment:
```zsh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notebooks

- **01_gbm_simulation.ipynb**: Simulates stock price paths using geometric Brownian motion
- **02_black_scholes_pricing.ipynb**: Calculates European option prices and delta values
- **03_portfolio_and_hedging.ipynb**: Visualizes hedging a long call position by shorting stock

## Notes

This is an early-stage project—documentation and features are still being developed.

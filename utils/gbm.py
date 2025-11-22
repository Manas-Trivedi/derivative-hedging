import numpy as np

def simulate_gbm_path(S0=100, mu=0.05, sigma=0.2, dt=1/252, steps=252):
    """Simulate one GBM price path."""
    prices = np.zeros(steps)
    prices[0] = S0
    for t in range(1, steps):
        Z = np.random.normal()
        prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return prices

def simulate_gbm_paths(S0=100, mu=0.05, sigma=0.2, dt=1/252, steps=252, n_paths=1000):
    """Simulate multiple GBM price paths."""
    all_paths = np.zeros((n_paths, steps))
    for i in range(n_paths):
        all_paths[i] = simulate_gbm_path(S0, mu, sigma, dt, steps)
    return all_paths
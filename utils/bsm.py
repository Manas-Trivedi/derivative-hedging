import numpy as np
from scipy.stats import norm

def bs_call_price(S, K, r, sigma, T):
    if(T <= 0):
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, r, sigma, T):
    if(T <= 0):
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    return delta_call, delta_put
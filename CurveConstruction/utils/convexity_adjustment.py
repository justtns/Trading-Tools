import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from fredapi import Fred
import pandas as pd

def sofr_volatility(API_KEY):
    #fred_api_key = '7304d4afcbd5fe7f4cb2399c18ec8267'  
    fred = Fred(api_key=API_KEY)
    sofr_overnight = fred.get_series('SOFR').last('3M')
    sofr_returns = sofr_overnight.pct_change().dropna()
    sofr_volatility = sofr_returns.std()
    return sofr_volatility

def local_volatility(sigma_F, t1, t2, F):
    return sigma_F / (1 + (t2 - t1) * (1 - F))

def bond_price_volatility(sigma_F, t1, t2, F, P):
    return (t2 - t1) * sigma_F * P * (1 - F)

def bond_price_approximation(F, t1, t2):
    return 1 / (1 + (t2 - t1) * (1 - F))

def convexity_adjustment(sigma, t1, t2):
    return (sigma ** 2) / 2 * (t2 - t1)
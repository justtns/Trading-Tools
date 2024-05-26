# Requirements:
# Install the required libraries using the following commands:
# pip install yfinance pandas numpy statsmodels 

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import requests

def is_valid_ticker(ticker):
    try:
        data = yf.download(ticker, period='1d')
        return not data.empty
    except Exception:
        return False

def get_ticker_from_name(input_string):
    if is_valid_ticker(input_string):
        return input_string
    else:
        try:
            res = requests.get(f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={input_string}&apikey=SZ83NMER0LTRQCI5')
            response = res.json()
            return response["bestMatches"][0]["1. symbol"]
        except Exception as e:
            print(f"Could not find ticker for {input_string}: {e}")
            return None

def get_price_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_lag = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag).fit()
    halflife = -np.log(2) / model.params[1]
    return halflife

def johansen_test(data, significance_level=0.05):
    result = coint_johansen(data, det_order=0, k_ar_diff=1)
    trace_stat = result.lr1
    trace_critical_values = result.cvt[:, int((1 - significance_level) * 100)]
    return trace_stat, trace_critical_values

def find_cointegrated_pairs(data):
    n = data.shape[1]
    keys = data.columns
    pairs = []
    half_lives = []
    
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            combined_data = pd.concat([S1, S2], axis=1)
            trace_stat, trace_critical_values = johansen_test(combined_data)
            if trace_stat[0] > trace_critical_values[0]: 
                pairs.append((keys[i], keys[j]))
                spread = S1 - S2
                half_life = calculate_half_life(spread)
                half_lives.append(half_life)
    
    return pairs, half_lives

def display_results(pairs, half_lives):
    results = pd.DataFrame({
        'Pair': pairs,
        'Half-life': half_lives
    })
    results.sort_values(by='Half-life', inplace=True)
    print(results.to_string(index=False))

if __name__ == "__main__":
    while True:
        asset_names = input("Enter asset names separated by commas: ").split(',')
        asset_names = [name.strip() for name in asset_names]
        if (len(asset_names) < 2):
            continue
        break
    while True:
        try:
            start_date = input("Enter start date (YYYY-MM-DD): ")
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        except:
            continue
        break
    
    tickers = []
    for name in asset_names:
        ticker = get_ticker_from_name(name)
        if ticker:
            tickers.append(ticker)
    if len(tickers) < 2:
        print("Assets not found! \n")
        quit(0)

    price_data = get_price_data(tickers, start_date, end_date)
    price_data.dropna(inplace=True)
    
    cointegrated_pairs, half_lives = find_cointegrated_pairs(price_data)
    print('Cointegrated pairs found:', len(cointegrated_pairs))
    
    display_results(cointegrated_pairs, half_lives)
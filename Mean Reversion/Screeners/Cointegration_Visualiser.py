
import numpy as np
import yfinance as yf
import json
import pandas as pd
import matplotlib.pyplot as plt

def get_price_data(data, ticker, start_date, end_date):      
    price_series = yf.download(ticker, start=start_date, end=end_date)[['Close']].rename(columns={"Close":ticker})
    data = pd.concat([data, price_series], axis=1)
    return data
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio * np.sqrt(252)  # Annualize the Sharpe ratio

with open('config.json', 'r') as f:
    config = json.load(f)

start_date = config['analysis']['start_date']
end_date = config['analysis']['end_date']
combo = "GEN,CSX".split(",")
eigen_weight = [0.32614806, 0.03502869]
data = pd.DataFrame([])
for ticker in combo:
        data = get_price_data(data, ticker, start_date, end_date)


cointegrated_portfolio = np.dot(data.values, eigen_weight)
z_score = (cointegrated_portfolio - np.mean(cointegrated_portfolio)) / np.std(cointegrated_portfolio)
position_size = -z_score    
returns = data.pct_change().dropna()   
strategy_returns = (returns.values * position_size[:-1, np.newaxis]).sum(axis=1)   
sharpe_ratio = calculate_sharpe_ratio(strategy_returns)   
cumulative_returns_strategy = np.cumsum(strategy_returns)    
cumulative_returns_assets = (1 + returns).cumprod() - 1    
max_cumulative_return = cumulative_returns_assets.max().max()
result = ({
            'combo': combo,
            'sharpe_ratio': sharpe_ratio,
            'cumulative_returns_strategy': cumulative_returns_strategy,
            'cointegrated_portfolio': cointegrated_portfolio,
            'max_cumulative_return_asset': max_cumulative_return
        })
print(f"Combination: {result['combo']}, Sharpe Ratio: {result['sharpe_ratio']}, Max Cumulative Return of an Asset: {result['max_cumulative_return_asset']}")
plt.plot(result['cumulative_returns_strategy'], label=f"Strategy: {result['combo']}")
plt.axhline(result['max_cumulative_return_asset'], linestyle='--', label=f"Max Asset: {result['combo']}")
plt.title('Cumulative Returns of the Strategies vs. Max Asset Return')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
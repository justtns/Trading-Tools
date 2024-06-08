import yfinance as yf
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_order
from statsmodels.regression.linear_model import OLS
import numpy as np
import warnings
import itertools
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
warnings.filterwarnings("ignore", category=UserWarning)

class portfolio_result:
    def __init__(self):
        self.trace = 0
        self.result_90 = 0
        self.result_95 = 0
        self.result_99 = 0
        self.eigen_weight = 0
    def display(self):
        print("Trace Statistics:", [self.trace])
        print("Critical Values (90%, 95%, 99%):", [self.result_90, self.result_95, self.result_99])
        print("Weights:", self.eigen_weight)

def calculate_half_life(y):
    y_lagged = np.roll(y, 1)
    y_lagged[0] = 0
    delta_y = y - y_lagged
    delta_y = delta_y[1:]
    y_lagged = y_lagged[1:]
    X = np.column_stack([y_lagged, np.ones(len(y_lagged))])
    regression_result = OLS(delta_y, X).fit()
    gamma = regression_result.params[0]
    half_life = -np.log(2) / gamma  
    return half_life

def get_price_data(data, ticker, start_date, end_date):      
    price_series = yf.download(ticker, start=start_date, end=end_date)[['Close']].rename(columns={"Close":ticker})
    data = pd.concat([data, price_series], axis=1)
    return data

def find_best_cointegration(data, test_stats, conf_intervals, eig):
    passing_rates = []
    for i in range(len(test_stats)):
        passing_rate = sum(test_stats[i] > conf_intervals[i])
        passing_rates.append(passing_rate)
    max_passing_rate = max(passing_rates)
    max_indices = [i for i, rate in enumerate(passing_rates) if rate == max_passing_rate]
    if len(max_indices) > 1:
        half_lives = []
        for indices in max_indices:
            half_lives.append(calculate_half_life( np.dot(data.values, eig[indices])))
        best_index = min(max_indices, key=lambda i: half_lives[i])
    else:
        best_index = max_indices[0]
    return best_index

def perform_johansen(data):
    try:
        lag_order = select_order(data, maxlags=10, deterministic='n')
        johansen_result = coint_johansen(data.values, det_order=0, k_ar_diff=lag_order.aic)
        index = find_best_cointegration(data, johansen_result.lr1, johansen_result.cvt, johansen_result.evec)   
        result = portfolio_result()
        result.trace = johansen_result.lr1[index]
        result.result_90 = johansen_result.cvt[index][0]
        result.result_95 = johansen_result.cvt[index][1]
        result.result_99 = johansen_result.cvt[index][2]
        result.eigen_weight = johansen_result.evec[index]
        return result
    except:
        return 0

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio * np.sqrt(252)  # Annualize the Sharpe ratio

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'ticker': ticker,
        'marketCap': info.get('marketCap'),
        'averageVolume': info.get('averageVolume')
    }

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)
    sp500_df = table[0]
    return sp500_df

def get_mid_large_cap_stocks(min_market_cap=2e8, min_avg_volume=1e6):
    sp500_tickers = get_sp500_tickers()['Symbol'].tolist()
    stock_data = []
    for ticker in sp500_tickers:
        info = get_stock_info(ticker)
        if info['marketCap'] and info['averageVolume']:
            if info['marketCap'] >= min_market_cap and info['averageVolume'] >= min_avg_volume:
                stock_data.append(info)
    return pd.DataFrame(stock_data)

def select_random_stocks(n=5, min_market_cap=2e8, min_avg_volume=1e6):
    stock_df = get_mid_large_cap_stocks(min_market_cap, min_avg_volume)
    return stock_df.sample(n=n)['ticker'].tolist()

def check_and_update_sheet(sheet, combination, sharpe_ratio, max_cumulative_return, portfolio):
    all_records = sheet.get_all_records()
    combination_str = ','.join(sorted(combination))

    for record in all_records:
        if sorted(record['Combination'].split(',')) == sorted(combination):
            print(f"Combination {combination_str} already exists in the sheet.")
            return
    
    # If combination doesn't exist, append the new data
    new_row = [combination_str, portfolio, sharpe_ratio, max_cumulative_return]
    sheet.append_row(new_row)
    print(f"Added new combination {combination_str} to the sheet.")

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
spreadsheet = client.open('Cointegration Screener')
sheet = spreadsheet.sheet1

with open('config.json', 'r') as f:
    config = json.load(f)

start_date = config['analysis']['start_date']
end_date = config['analysis']['end_date']
num_stocks = config['analysis']['num_stocks']
min_market_cap = config['market_criteria']['min_market_cap']
min_avg_volume = config['market_criteria']['min_avg_volume']

while True:
    tickers = select_random_stocks(num_stocks, min_market_cap, min_avg_volume)
    # Store the results
    all_data = pd.DataFrame([])
    for ticker in tickers:
        all_data = get_price_data(all_data, ticker, start_date, end_date)

    results = []
    for i in range(2,len(tickers)+1):
        ticker_combinations = list(itertools.combinations(tickers, i))
        for combo in ticker_combinations:
            data = all_data[list(combo)]
            print(f"Testing combination: {combo}")
            result = perform_johansen(data)
            if result == 0:
                print("Combination failed to converge.\n")
                continue
            cointegrated_portfolio = np.dot(data.values, result.eigen_weight)
            z_score = (cointegrated_portfolio - np.mean(cointegrated_portfolio)) / np.std(cointegrated_portfolio)
            position_size = -z_score    
            returns = data.pct_change().dropna()   
            strategy_returns = (returns.values * position_size[:-1, np.newaxis]).sum(axis=1)   
            sharpe_ratio = calculate_sharpe_ratio(strategy_returns)   
            cumulative_returns_strategy = np.cumsum(strategy_returns)    
            cumulative_returns_assets = (1 + returns).cumprod() - 1    
            max_cumulative_return = cumulative_returns_assets.max().max()
            if sharpe_ratio > 1:
                results.append({
                    'combo': combo,
                    'sharpe_ratio': sharpe_ratio,
                    'cumulative_returns_strategy': cumulative_returns_strategy,
                    'cointegrated_portfolio': cointegrated_portfolio,
                    'max_cumulative_return_asset': max_cumulative_return
                })
    for result in results:
        combo_str = ','.join(sorted(result['combo']))
        cointegrated_portfolio = result['cointegrated_portfolio']
        sharpe_ratio = result['sharpe_ratio']
        max_cumulative_return = result['max_cumulative_return_asset']
        check_and_update_sheet(sheet, result['combo'], sharpe_ratio, max_cumulative_return, cointegrated_portfolio)
# Import necessary libraries
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta
import locale
import math
import copy
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from fredapi import Fred
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
import json

# Based on: https://www.banqueducanada.ca/wp-content/uploads/2010/01/wp00-17.pdf
# Futures Convexity: https://timxiao1203.github.io/FuturesConvexity.pdf

# Function to calculate zero swap rate for the short end of the curve
def get_zero_swap_rate_short(r, m, y=360):
    rc = [(y/m[i]) * np.log(1 + r[i]/(y/m[i])) for i in range(len(m))]
    return rc

# Function to calculate days to maturity
def get_days_to_maturity(cur_date, rateMaturities):
    return [(rateMaturities[x] - cur_date).days for x in range(len(rateMaturities))]

# Function to calculate discount factor
def get_disc_factor(days, rates):
    return [np.exp(-1 * days[x] / 360.0 * rates[x]) for x in range(len(days))]

# Function to obtain SOFR volatility from Fred API
def sofr_volatility(API_KEY):
    fred = Fred(api_key=API_KEY)
    sofr_overnight = fred.get_series('SOFR').last('3M')
    sofr_returns = sofr_overnight.pct_change().dropna()
    return sofr_returns.std()

# Helper functions for convexity adjustment
def local_volatility(sigma_B, P, F):
    return sigma_B / (P * (1 - F))

def bond_price_volatility(sigma_S, duration, F, P):
    return duration * sigma_S * P * (1 - F)

def bond_price_approximation(F, duration):
    return 1 / (1 + duration * (1 - F))

def convexity_adjustment(sigma, duration):
    return (sigma ** 2) / 2 * duration

# Function to bootstrap zero rates from swap rates
def bootstrap_zero_rate(T, coupon_rate, previous_zero_rates, m=2):
    discounted_sum = sum((coupon_rate / m) * np.exp(-previous_zero_rates[i / m] * i / m) 
                         for i in range(1, int(T * m)) if i / m in previous_zero_rates)
    num = 1 - discounted_sum
    denom = 1 + (coupon_rate / m)
    return -np.log(num / denom) / T

# Step 1: Retrieve SOFR Rates and calculate the short end of the swap curve
with open('credentials.json', 'r') as f:
    fred_api_key = json.load(f)['FRED_API']
fred = Fred(api_key=fred_api_key)
sofr_overnight = fred.get_series('SOFR').last('1d').values[0] / 100
cur_date = datetime.date.today()

rate = [sofr_overnight, 5.118 / 100, 4.94607 / 100]
rate_maturities = [cur_date + timedelta(days=1), 
                   cur_date + relativedelta(months=1), 
                   cur_date + relativedelta(months=3)]

days_to_maturity_short = get_days_to_maturity(cur_date, rate_maturities)
zero_rates_deposit_short = get_zero_swap_rate_short(rate, days_to_maturity_short)
discounts_short = get_disc_factor(days_to_maturity_short, zero_rates_deposit_short)

df_short = pd.DataFrame({
    'Maturities': pd.Series(rate_maturities),
    'Number_of_Days': pd.Series(days_to_maturity_short),
    'Original_Rates': pd.Series(rate),
    'Zero_Rates': pd.Series(zero_rates_deposit_short),
    'Discount_Rates': pd.Series(discounts_short),
    'Instrument': 'Deposit'
})

# Step 2: Calculate futures-based swap curve with convexity adjustment
future_prices = {
    'Date': ['2025-01-01', '2025-06-01', '2025-12-01', '2026-03-01', '2026-09-01'],
    'Price': [96.08, 96.6, 96.89, 96.95, 96.98]
}
df_medium = pd.DataFrame(future_prices)
df_medium['Date'] = pd.to_datetime(df_medium['Date'])
df_medium['Date_3_Months_After'] = df_medium['Date'] + pd.DateOffset(months=3)

sofr_30_vol = sofr_volatility(fred_api_key)

df_medium['Approximated_Bond_Price'] = df_medium.apply(
    lambda row: bond_price_approximation(row['Price'] / 100, 
                                         (row['Date_3_Months_After'] - row['Date']).days / 360), axis=1
)

df_medium['Bond_Volatility'] = df_medium.apply(
    lambda row: bond_price_volatility(sofr_30_vol, 
                                      (row['Date_3_Months_After'] - row['Date']).days / 360, 
                                      row['Price'] / 100, 
                                      row['Approximated_Bond_Price']), axis=1
)

df_medium['Futures_Volatility'] = df_medium.apply(
    lambda row: local_volatility(row['Bond_Volatility'], 
                                 row['Approximated_Bond_Price'], 
                                 row['Price'] / 100), axis=1
)

df_medium['Convex_Adj'] = df_medium.apply(
    lambda row: 100 * convexity_adjustment(row['Futures_Volatility'], 
                                           (row['Date_3_Months_After'] - row['Date']).days / 360), axis=1
)

df_medium['Original_Rates'] = (100 - df_medium['Price']) / 100
df_medium['Adjusted_Futures_Rate'] = df_medium['Original_Rates'] + df_medium['Convex_Adj']
df_medium['Days_To_Maturity'] = get_days_to_maturity(pd.Timestamp(cur_date), df_medium['Date_3_Months_After'])

df_medium['Adjusted_Futures_Rate'] = get_zero_swap_rate_short(df_medium['Adjusted_Futures_Rate'].to_list(), df_medium['Days_To_Maturity'].to_list())
df_medium['Discount_Rates'] = 0
df_medium['Instrument'] = "Futures"

df_medium = df_medium[['Date', 'Days_To_Maturity', 'Instrument', 'Original_Rates', 'Adjusted_Futures_Rate', 'Discount_Rates']].rename(columns={
    'Days_To_Maturity': 'Number_of_Days',
    'Adjusted_Futures_Rate': 'Zero_Rates',
    'Date': 'Maturities'
})

# Step 3: IRS Par Rates
swap_par_rates = {
    'Years': [1, 3, 5, 8, 10, 15, 20],
    'Original_Rates': [4.3925 / 100, 3.6741 / 100, 3.5562 / 100, 3.565 / 100, 3.5965 / 100, 3.669 / 100, 3.63 / 100]
}
df_long = pd.DataFrame(swap_par_rates)

zero_rates = []
previous_zero_rates = {}
for idx, row in df_long.iterrows():
    T = row['Years']
    coupon_rate = row['Original_Rates']
    zero_rate_T = bootstrap_zero_rate(T, coupon_rate, previous_zero_rates)
    zero_rates.append(zero_rate_T)
    previous_zero_rates[T] = zero_rate_T

df_long['Maturities'] = pd.to_datetime(cur_date) + pd.to_timedelta(df_long['Years'] * 365, unit='D')
df_long['Zero_Rates'] = zero_rates
df_long['Number_of_Days'] = df_long['Years'] * 365
df_long['Instrument'] = "IRS"
df_long['Discount_Rates'] = get_disc_factor(df_long['Number_of_Days'], df_long['Zero_Rates'])

# Combine all dataframes
rates = pd.concat([df_short, df_medium, df_long])
rates['Maturities'] = pd.to_datetime(rates['Maturities'], errors='coerce')
rates['Maturities'] = rates['Maturities'].dt.date
rates = rates.set_index("Maturities")

# Interpolation - Cubic Spline
zero_rates = rates['Zero_Rates'].values
maturities = rates['Number_of_Days'].values
cubic_spline = CubicSpline(maturities, zero_rates, bc_type='natural')
maturities_interpolated = np.linspace(min(maturities), max(maturities), 500)
interpolated_zero_rates = cubic_spline(maturities_interpolated)

# Plot cubic spline
plt.figure(figsize=(10, 6))
plt.plot(maturities, zero_rates, 'o', label='Original Zero Rates', color='red')
plt.plot(maturities_interpolated, interpolated_zero_rates, label='Interpolated Swap Curve (Cubic Spline)', color='blue')
plt.xlabel('Days to Maturity')
plt.ylabel('Zero Rate')
plt.title('Piecewise Cubic Spline Interpolation of Swap Curve')
plt.legend()
plt.grid(True)
plt.show()

# Interpolation - Linear
linear_interpolation = interp1d(maturities, zero_rates, kind='linear', fill_value="extrapolate")
interpolated_zero_rates = linear_interpolation(maturities_interpolated)

# Plot linear interpolation
plt.figure(figsize=(10, 6))
plt.plot(maturities, zero_rates, 'o', label='Original Zero Rates', color='red')
plt.plot(maturities_interpolated, interpolated_zero_rates, label='Interpolated Swap Curve (Piecewise Linear)', color='blue')
plt.xlabel('Days to Maturity')
plt.ylabel('Zero Rate')
plt.title('Piecewise Linear Interpolation of Swap Curve')
plt.legend()
plt.grid(True)
plt.show()

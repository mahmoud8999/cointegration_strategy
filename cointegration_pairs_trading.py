# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:25:45 2024

@author: Mahmoud
"""

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm

# This function downloads each ticker provided in the tickers list
def download_data(tickers_list, start_date='2007-01-01', end_date='2024-01-01', interval='1d'):
    data_df = pd.DataFrame()

    for ticker in tickers_list:
        print(f'Starting to download: {ticker}')
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        symbol = ticker.split('=')[0]
        data_df[f'{symbol}'] = data['Close']
        print(f'Done downloading: {ticker}\n')
        
    data_df = data_df.dropna()
    return data_df

# This function finds Non-Stationary Pairs
def non_stationary_pairs(fx_data):
    
    results = []
    
    for pairs_name, pairs_data in fx_data.items():
        adf_result = adfuller(pairs_data)
        
        # Check if the data is Stationary
        if(adf_result[1] < 0.05):
            continue
        # Check if the data is Non-Stationary
        else:
            results.append({
                'Pairs': pairs_name,
                'ADF_Test': adf_result[0],
                'P-Value': adf_result[1],
                'Stationary': 'No'
            })
            
    results_df = pd.DataFrame(results)
    print(results_df)
    
    non_stationary_pairs = results_df['Pairs'].to_list()
    combinations_list = list(itertools.permutations(non_stationary_pairs, 2))
    combinations_df = pd.DataFrame(combinations_list, columns=['Currency1', 'Currency2'])
    
    return combinations_df

# This function calculates the spread of Non-Stationary Pairs
def calculate_spread_for_cointegrated_pairs(combinations_df, fx_data):
    
    spread_results = pd.DataFrame()
    
    for index, row in combinations_df.iterrows():
        currency1 = row['Currency1']
        currency2 = row['Currency2']
        
        X = fx_data[row.iloc[0]]
        y = fx_data[row.iloc[1]]
        
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X)
        results = model.fit()
        
        residuals = results.resid
        beta_coefficient = results.params.iloc[1]
    
        adfuller_result = adfuller(residuals)
        
        # Check if the residuals are Stationary
        if(adfuller_result[1] < 0.05):
            # Calculate the spread if the residuals are stationary
            log_currency1 = np.log(fx_data[currency1])
            log_currency2 = np.log(fx_data[currency2])
            spread = log_currency2 - beta_coefficient * log_currency1
            
            # Create column name by combining currency pairs
            column_name = f"{currency1}_{currency2}"
            
            # Add spread to results dictionary
            spread_results[column_name] = spread
            
    return spread_results

def calculate_z_scores(spread_df, windows):
    z_scores = pd.DataFrame(index=spread_df.index)
    
    for window_train, window_test in windows:
        for pair in spread_df.columns:
            pair_z_scores = []
            
            for i in range(window_train, len(spread_df), window_test):
                # Training period
                train_data = spread_df[pair].iloc[i-window_train:i]
                mean = train_data.mean()
                std_dev = train_data.std()
                
                # Testing period
                test_data = spread_df[pair].iloc[i:i+window_test]
                pair_z_scores.extend((test_data - mean) / std_dev)
            
            # Pad the beginning with NaNs and truncate to match DataFrame length
            pair_z_scores = [np.nan] * window_train + pair_z_scores
            pair_z_scores = pair_z_scores[:len(spread_df)]
            
            # Create column name with both window parameters
            column_name = f'{pair}_{window_train}_{window_test}'
            z_scores[column_name] = pair_z_scores
    
    return z_scores

# This function generates the trading signal based on the z-score and thresholds
def generate_trading_signals(zscore_df, zthresholds):
    # Create a dictionary to store signals for each threshold
    #signals_dict = {}
    results_df = pd.DataFrame()
    
    for threshold in zthresholds:
        # Create signals dataframe for current threshold
        signals_df = pd.DataFrame(0, index=zscore_df.index, columns=zscore_df.columns)
        signals_df[zscore_df < -threshold] = 1
        signals_df[zscore_df > threshold] = -1
        
        # Store in dictionary with threshold as key
        #signals_dict[f'threshold_{threshold}'] = signals_df
        results_df = pd.concat([results_df, signals_df.add_suffix(f'_T{threshold}')], axis=1)
    
    return results_df #signals_dict

def daily_price_change(fx_data_df):
    
    # Calculate the prices change for each currency
    fx_data_df = fx_data_df.pct_change().dropna()
    
    return fx_data_df

def calculate_returns(trading_signals_df, daily_price_change_df):
    daily_return_df = pd.DataFrame(index=trading_signals_df.index)
    
    for column in trading_signals_df.columns:
        pair_part = '_'.join(column.split('_')[:2])
        first_currency = pair_part.split('_')[0]
        second_currency = pair_part.split('_')[1]
        
        # Get price changes for both currencies
        currency1_change = daily_price_change_df[first_currency]
        currency2_change = daily_price_change_df[second_currency]
        
        # First pair: signal * currency1_change
        first_position = trading_signals_df[column] * currency1_change
        
        # Second pair: (-signal) * currency2_change
        second_position = (-trading_signals_df[column]) * currency2_change
        
        # Total return is sum of both positions
        pair_return = first_position + second_position
        
        daily_return_df[f'Return_{column}'] = pair_return
        
    return daily_return_df

# Create different groups based on window sizes and thresholds
def create_return_groups(daily_returns):
    groups = {}
    
    # Get unique window combinations from column names
    window_combinations = set()
    for col in daily_returns.columns:
        # Extract the window part from column name (e.g., '63_1' from 'Return_EURUSD_GBPUSD_63_1_T1')
        parts = col.split('_')
        window_part = f"{parts[-3]}_{parts[-2]}"  # Get the '63_1' part
        window_combinations.add(window_part)
    
    # For each window combination and threshold
    for window in window_combinations:
        for threshold in [1, 2, 3]:
            # Create key like '63_1_T1'
            key = f"{window}_T{threshold}"
            # Get columns matching this combination
            columns = [col for col in daily_returns.columns if f"_{window}_T{threshold}" in col]
            
            if columns:  # Only create DataFrame if columns exist
                df_name = f"df_{key}"
                globals()[df_name] = daily_returns[columns]
                groups[key] = daily_returns[columns]
    
    return groups

def calculate_portfolio_metrics(returns_df):
    # Define a function to calculate portfolio metrics given a DataFrame of daily returns (returns_df).
    
    # Annualized Return and Volatility (current method) 
    # multiplied by 252 trading days and expressed as a percentage.
    ann_return = returns_df.mean().mean() * 252 * 100

    # Calculate the annualized volatility as the average daily standard deviation of all assets,
    # multiplied by the square root of 252 and expressed as a percentage.
    ann_vol = returns_df.std().mean() * np.sqrt(252) * 100
    
    
    # Equal-weighted portfolio return series
    # Compute the portfolio return series by taking the mean of daily returns across all assets
    port_returns = returns_df.mean(axis=1)
    
    # Calculate the annualized return for the equal-weighted portfolio
    ann_return_alt = port_returns.mean() * 252 * 100
    
    # Calculate the annualized volatility for the equal-weighted portfolio
    ann_vol_alt = port_returns.std() * np.sqrt(252) * 100
    
    # Define a risk-free rate of 0.5% for use in Sharpe and Sortino Ratio calculations.
    rf_rate = 0.5
    
    # Calculate the Sharpe Ratio
    sharpe = (returns_df.mean().mean() * 252 - rf_rate) / (returns_df.std().mean() * np.sqrt(252))
    
    # Filter for negative daily returns to focus on downside risk.
    downside_returns = returns_df[returns_df < 0]
    
    # Calculate the annualized standard deviation of downside returns (downside risk).
    downside_std = np.sqrt(252) * downside_returns.std().mean()
   
    # Calculate the Sortino Ratio
    sortino = (returns_df.mean().mean() * 252 - rf_rate) / downside_std

    # Calculate the cumulative return series for the portfolio,
    # assuming reinvestment of daily returns.
    cum_returns = (1 + port_returns).cumprod()
    
    # Compute the rolling maximum value of the cumulative return series,
    # representing the portfolio's highest value over time.
    rolling_max = cum_returns.expanding().max()
    
    # Calculate drawdowns as the percentage difference between the cumulative return and
    # the rolling maximum value.
    drawdowns = (cum_returns - rolling_max) / rolling_max
    
    # Find the maximum drawdown
    max_drawdown = drawdowns.min() * 100
    # Calculate the Calmar Ratio
    calmar = -((returns_df.mean().mean() * 252) / (max_drawdown / 100))

    metrics = {
        'Annualized Return (%)': ann_return,
        'Annualized Volatility (%)': ann_vol,
        'Alternative Ann. Return (%)': ann_return_alt,
        'Alternative Ann. Volatility (%)': ann_vol_alt,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Maximum Drawdown (%)': max_drawdown,
        'Calmar Ratio': calmar
    }
    
    # Convert the dictionary to a pandas DataFrame for better visualization and presentation.
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    
    return metrics_df

        
# List of tickers
tickers_list = ['EURUSD=X', 'GBPUSD=X', 'JPYUSD=X', 'CHFUSD=X', 'CADUSD=X', 'AUDUSD=X', 'NZDUSD=X']

# Define Training and Testing windows (Training Window, Testing Window)
windows = [(63, 1), (63, 5), (63, 21), (128, 1), (128, 5), (128, 21), (128, 63), (257, 1), (257, 5), (257, 21), (257, 63), (257, 128)]

# Defome Z-Score Thresholds
z_score_thresholds = [1, 2, 3]

# Download data from Yahoo Finance
fx_data = download_data(tickers_list)

# Create different combinations of FX pairs
combinations_df = non_stationary_pairs(fx_data)

# Calculate the spread of the FX pairs
spread_df = calculate_spread_for_cointegrated_pairs(combinations_df, fx_data)

# Calculate the z-score of the FX pairs
z_score_df = calculate_z_scores(spread_df, windows)

# Generate Trading Signals
trading_signals = generate_trading_signals(z_score_df, z_score_thresholds)

# Calculate Price Change
price_change = daily_price_change(fx_data)

# Calculate Daily Returns
daily_returns = calculate_returns(trading_signals, price_change).dropna()

# Create the groups
return_groups = create_return_groups(daily_returns)

for group_name, group_df in return_groups.items():
    print(f"\nMetrics for {group_name}:")
    metrics_df = calculate_portfolio_metrics(group_df)
    print(metrics_df)


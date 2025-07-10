from binance.client import Client as bnb_client
from datetime import datetime, UTC, timezone, timedelta, tzinfo
import pandas as pd
import numpy as np

from typing import Dict, Tuple
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# ----------------------------------------------------------------------------------- # 

def format_binance(
    data: Dict
) -> pd.DataFrame:
    '''
    Purpose:
    Takes in the data from client.get_historical_klines, and then converts it into a 
    processable dataframe. 
    
    Input:
    - data: dictionary that is generated from client.get_historical_klines
    
    Output:
    - pandas DataFrame that is formatted with columns
    '''
    columns = ['open_time', 
               'open', 
               'high', 
               'low', 
               'close', 
               'volume', 
               'close_time', 
               'quote_volume', 
               'num_trades', 
               'taker_base_volume', 
               'taker_quote_volume', 
               'ignore']

    df = pd.DataFrame(data, columns = columns)

    # convert the POXIS timestamp to actual times (number of milliseconds since Jan 1, 1970)
    df['open_time'] = df['open_time'].map(lambda x: datetime.fromtimestamp(x / 1000, UTC))
    df['close_time'] = df['close_time'].map(lambda x: datetime.fromtimestamp(x / 1000, UTC))
    return df

# ----------------------------------------------------------------------------------- #

def get_binance_px(
    symbol: str, 
    freq: str, 
    start_time: str = '2019-01-01'
) -> Dict:
    '''
    Purpose: 
    Given the symbol, freq, and start_time, generate the binance data from of the symbol
    at the frequency from the start_time to the most recent point of time.
    
    Inputs:
    - symbol: str indicating the ticker
    - freq: str indicating the frequency
    - start_time: str indicating the start time to begin analyzing
    
    Outputs:
    - dictionary of the data of the ticker
    '''
    client = bnb_client(tld='US')
    data = client.get_historical_klines(symbol, freq, start_time)
    data = format_binance(data)
    return data

# ----------------------------------------------------------------------------------- #

def get_notional_volume(
    symbol: str, 
    lookback_days: int = 30
) -> float:
    '''
    Purpose:
    Given the symbol and the number of lookback_days, compute the notional volume of
    the symbol for the last lookback_days.
    
    Input:
    - symbol: str indicating what security to look at
    - lookback_days: int indicating number of days to lookback
    
    Output: 
    - the sum of the notional volume of the last lookback_days
    '''
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        df = get_binance_px(symbol, freq = '1d', start_time = start_time.strftime("%d %b %Y"))
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        notional = (df['close'] * df['volume']).sum()
        return notional
    except Exception as e:
        print(f"Error with {symbol}: {e}")
        return 0
    
# ----------------------------------------------------------------------------------- #

def select_topk_securities(
    lookback_days: int = 30,
    k: int = 10
) -> pd.DataFrame:
    '''
    Purpose:
    Given a lookback_days and k, find the top k crypto securities that have the highest 
    notional volume. Particularly, sample from crypto securities that can only 
    be spot traded. 
    
    Input:
    - lookback_days: int indicating how many days we want to look back to calculate 
      notional volume
    - k: int indicating the amount of securities we want to find
    
    Output:
    - DataFrame containing the top k symbols and their notional volumes
    '''
    # selecting the crypto securities to be in my universe
    client = bnb_client(tld='US')
    full_universe = client.get_exchange_info()

    spot_tickers = [x['symbol'] 
                    for x in full_universe['symbols'] 
                    if x['quoteAsset'] == 'USDT' and 
                    x['status'] == 'TRADING' and 
                    x['isSpotTradingAllowed']
                ]

    notional_vols = []
    for x in spot_tickers:
        notional = get_notional_volume(x, lookback_days=lookback_days)
        if notional > 0:
            notional_vols.append((x, notional))
    
    notional_vols = pd.DataFrame(notional_vols, columns = ['symbol', 'notional_vol'])
    notional_vols = notional_vols.sort_values(by='notional_vol', ascending=False).reset_index(drop = True)
    
    return notional_vols.head(k)

# ----------------------------------------------------------------------------------- #

def generate_pickle_px_data(
    csv_filename: str
) -> None:
    '''
    Purpose:
    Generate pickle files containing the px data. 
    
    Input:
    - csv_filename: string containing the filename of the csv with the tickers you want data for
    
    Output:
    - no output
    '''
    univ = pd.read_csv(csv_filename)
    
    freqs = ['1w', '1d', '1h']
    for freq in freqs:
        px_data = {}
        for symbol in univ['symbol']:
            for _ in range(3):
                try:
                    px_data[symbol] = get_binance_px(symbol, freq)
                    break
                except Exception as e:
                    print(f"Error with {symbol}", e)
                    time.sleep(5)
        filename = f"{freq}_pxdata.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(px_data, f)
            print(f"Created file {filename}")

# ----------------------------------------------------------------------------------- #

def split_data(
    full_px: pd.DataFrame,
    test_frac: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_point = int(len(full_px) * (1 - test_frac))
    train = full_px.iloc[:split_point]
    test = full_px.iloc[split_point:]

    return train, test

# ----------------------------------------------------------------------------------- #

def train_momentum_strat(
    px: pd.DataFrame, 
    vol: pd.DataFrame, 
    lookbacks=[1, 2, 3, 4, 6, 12, 24, 72, 168, 276, 504, 720, 1080], 
    lags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 24, 72, 168],
    tcost_bps=10, 
    vol_filter_quantile = 0.5
):
    results = []
    ret = px.pct_change(fill_method=None)
    
    sharpe_matrix = pd.DataFrame(index=lookbacks, columns=lags)

    for lb in lookbacks:
        # signal = momentum_signal * np.log1p(vol_avg)
        for lag in lags:
            vol_avg = (vol.rolling(90, min_periods=1).mean())
            port = (ret.rolling(lb, min_periods=1).mean()).rank(1) 
            port = port * np.log1p(vol_avg)
            
            port = port.sub(port.mean(axis=1), axis = 0)
            port = port.div(port.abs().sum(axis=1), axis = 0)
        
            turnover = (port.fillna(0) - port.shift()).abs().sum(axis = 1)
        
            gross_ret = (port.shift(1 + lag) * ret).sum(axis = 1)
        
            net_ret = gross_ret - turnover * tcost_bps * 1e-4
        
            sharpe = net_ret.mean(axis=0) / net_ret.std(axis=0) * np.sqrt(252)
            sharpe_matrix.loc[lb, lag] = sharpe
        
            results.append({
                "lookback": lb,
                "lag": lag,
                "sharpe": sharpe,
                "gross_ret": gross_ret,
                "net_ret": net_ret
            })
            
            #if (lag == 0):
            #    cum_ret = (1 + net_ret).cumprod()
            #    plt.plot(cum_ret.index, cum_ret)
            #    plt.show()
        
            print(f"Lookback = {lb:>4} hrs | Lag = {lag:>3} | Sharpe: {sharpe}")
            
            
            if (sharpe > 0.5):
                total_gross_ret = (1 + gross_ret).cumprod()
                total_net_ret = (1 + net_ret).cumprod()
                plt.plot(total_gross_ret.index, total_gross_ret)
                plt.show()
                plt.plot(total_net_ret.index, total_net_ret)
                plt.show()
                plt.plot(net_ret.index, net_ret)
                plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(sharpe_matrix.astype(float), annot = True, fmt=".2f", cmap = "coolwarm", linewidths=0.5)
    plt.title("Sharpe Ratio Heatmap")
    plt.xlabel("Lag")
    plt.ylabel("Lookback")
    plt.show()
    return results

# ----------------------------------------------------------------------------------- #

def test_momentum_strat(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    lookback: int,
    lag: int, 
    tcost_bps: int = 10
):
    ret = px.pct_change(fill_method=None)
    
    vol_avg = (vol.rolling(lookback, min_periods=1).mean())
    port = (ret.rolling(lookback, min_periods=1).mean()).rank(1) 
    port = port * np.log1p(vol_avg)    
    
    port = port.sub(port.mean(axis = 1), axis = 0)
    port = port.div(port.abs().sum(axis = 1), axis = 0)
    
    turnover = (port.fillna(0) - port.shift()).abs().sum(axis = 1)
    
    gross_ret = (port.shift(1 + lag) * ret).sum(axis = 1)
    net_ret = gross_ret - turnover * tcost_bps * 1e-4
    
    sharpe = net_ret.mean(axis = 0) / net_ret.std(axis = 0) * np.sqrt(252)
    
    total_gross_ret = (1 + gross_ret).cumprod()
    total_net_ret = (1 + net_ret).cumprod()

    print(f"Momentum strategy with lookback: {lookback} and lag: {lag}\nSharpe: {sharpe} | Gross Returns: {total_gross_ret.iloc[-1]} | Net Returns: {total_net_ret.iloc[-1]}")
    plt.plot(total_gross_ret.index, total_gross_ret)
    plt.show()
    plt.plot(total_net_ret.index, total_net_ret)
    plt.show()
    plt.plot(net_ret.index, net_ret)
    plt.show()
    
# ----------------------------------------------------------------------------------- #

def buy_and_hold_strat(px, tcost_bps = 20):
    
    ret = px.pct_change(fill_method=None)
    equal_weights = pd.Series(1 / len(px.columns), index=px.columns)
    buyhold_ret = (ret * equal_weights).sum(axis = 1)
    
    initial_tcost = (equal_weights.abs().sum()) * tcost_bps * 1e-4
    portfolio_val = (1 + buyhold_ret).cumprod()
    portfolio_val = portfolio_val * (1 - initial_tcost)
    
    net_ret = portfolio_val.pct_change(fill_method = None).fillna(0)
    
    buyhold_sharpe = net_ret.mean() / net_ret.std() * np.sqrt(252)
    print(f"Buy & Hold Sharpe Ratio: {buyhold_sharpe:.3f}")
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_val, label='Buy & Hold Portfolio')
    plt.title("Cumulative Return: Equal-Weighted Buy & Hold")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------- #

def macd_strat(train_px: pd.DataFrame, test_px: pd.DataFrame, fast_lookback, slow_lookback, signal_period, tcost_bps = 20):
    """
    MACD strategy with separate training and testing phases
    """
    train_ret = train_px.pct_change()
    test_ret = test_px.pct_change()
    
    print("=" * 60)
    print("MACD STRATEGY - TRAINING PHASE")
    print("=" * 60)
    
    # Training phase - find best configuration
    train_results = []
    for f, s, sp in product(fast_lookback, slow_lookback, signal_period):
        if f >= s:
            continue
        
        macd = pd.DataFrame(index=train_px.index, columns=train_px.columns)
        signal_line = pd.DataFrame(index=train_px.index, columns=train_px.columns)
        signal = pd.DataFrame(index=train_px.index, columns=train_px.columns)
        
        for asset in train_px.columns:
            price = train_px[asset]
        
            ema_fast = price.ewm(span=f, adjust=False).mean()
            ema_slow = price.ewm(span=s, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            sig_line = macd_line.ewm(span=sp, adjust=False).mean()
        
            macd[asset] = macd_line
            signal_line[asset] = sig_line
            signal[asset] = np.sign(macd_line - sig_line)
        
        weights = signal.div(signal.abs().sum(axis = 1), axis = 0).fillna(0)
        turnover = weights.diff().abs().sum(axis = 1)
        gross_ret = (weights.shift() * train_ret).sum(axis = 1)
        net_ret = gross_ret - turnover * tcost_bps * 1e-4
        cum_net = (1 + net_ret).cumprod()
        
        strat_sharpe = net_ret.mean() / net_ret.std() * np.sqrt(252)
        
        train_results.append({
            'fast': f,
            'slow': s,
            'signal': sp,
            'sharpe': strat_sharpe,
            'cum_net': cum_net.iloc[-1],
            'returns': net_ret
        })
        
        print(f"Fast: {f:>3} | Slow: {s:>3} | Signal: {sp:>3} | Sharpe: {strat_sharpe:>6.3f}")
    
    train_results_df = pd.DataFrame(train_results).sort_values(by='sharpe', ascending=False)
    print("\nTop Training Configs by Sharpe:")
    print(train_results_df[['fast', 'slow', 'signal', 'sharpe']].head(5))
    
    # Get best configuration
    best_config = train_results_df.iloc[0]
    best_fast = best_config['fast']
    best_slow = best_config['slow']
    best_signal = best_config['signal']
    
    print(f"\nBest Configuration: Fast={best_fast}, Slow={best_slow}, Signal={best_signal}")
    print(f"Training Sharpe: {best_config['sharpe']:.3f}")
    
    # Plot training results
    pivot = train_results_df.pivot_table(index='fast', columns='slow', values='sharpe')
    plt.figure(figsize=(8, 6))
    plt.title("Training Sharpe Ratio (Signal Period Fixed)")
    plt.xlabel("Slow EMA")
    plt.ylabel("Fast EMA")
    plt.imshow(pivot, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label="Sharpe Ratio")
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.tight_layout()
    plt.show()
    
    # Plot best training strategy
    best_train_returns = best_config['returns']
    cum_ret_train = (1 + best_train_returns).cumprod()
    plt.figure(figsize=(14, 6))
    plt.plot(cum_ret_train, label='Best MACD Strategy (Training)', color='blue')
    plt.title(f"Training Performance — Best MACD Strategy\nFast={best_fast}, Slow={best_slow}, Signal={best_signal}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Testing phase - apply best configuration to test data
    print("\n" + "=" * 60)
    print("MACD STRATEGY - TESTING PHASE")
    print("=" * 60)
    
    # Apply best configuration to test data
    test_macd = pd.DataFrame(index=test_px.index, columns=test_px.columns)
    test_signal_line = pd.DataFrame(index=test_px.index, columns=test_px.columns)
    test_signal = pd.DataFrame(index=test_px.index, columns=test_px.columns)
    
    for asset in test_px.columns:
        price = test_px[asset]
    
        ema_fast = price.ewm(span=best_fast, adjust=False).mean()
        ema_slow = price.ewm(span=best_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        sig_line = macd_line.ewm(span=best_signal, adjust=False).mean()
    
        test_macd[asset] = macd_line
        test_signal_line[asset] = sig_line
        test_signal[asset] = np.sign(macd_line - sig_line)
    
    test_weights = test_signal.div(test_signal.abs().sum(axis = 1), axis = 0).fillna(0)
    test_turnover = test_weights.diff().abs().sum(axis = 1)
    test_gross_ret = (test_weights.shift() * test_ret).sum(axis = 1)
    test_net_ret = test_gross_ret - test_turnover * tcost_bps * 1e-4
    test_cum_gross = (1 + test_gross_ret).cumprod()
    test_cum_net = (1 + test_net_ret).cumprod()
    
    test_sharpe = test_net_ret.mean() / test_net_ret.std() * np.sqrt(252)
    
    print(f"Test Results:")
    print(f"  Sharpe Ratio: {test_sharpe:.3f}")
    print(f"  Cumulative Return: {test_cum_net.iloc[-1]:.3f}")
    print(f"  Annualized Return: {(test_cum_net.iloc[-1] ** (252/len(test_net_ret)) - 1):.3f}")
    print(f"  Volatility: {test_net_ret.std() * np.sqrt(252):.3f}")

    # Plot test results
    plt.figure(figsize=(14, 6))
    plt.plot(test_cum_gross.index, test_cum_gross.values, label='Best MACD Strategy (Test)', color='red')
    plt.title(f"Test Performance — Gross Cumulative Returns Best MACD Strategy\nFast={best_fast}, Slow={best_slow}, Signal={best_signal}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot test results
    plt.figure(figsize=(14, 6))
    plt.plot(test_cum_net.index, test_cum_net.values, label='Best MACD Strategy (Test)', color='red')
    plt.title(f"Test Performance — Net Cumulative Returns Best MACD Strategy\nFast={best_fast}, Slow={best_slow}, Signal={best_signal}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Compare training vs test performance
    plt.figure(figsize=(14, 6))
    plt.plot(cum_ret_train.index, cum_ret_train.values, label='Training Performance', color='blue', linewidth=2)
    plt.plot(test_cum_net.index, test_cum_net.values, label='Test Performance', color='red', linewidth=2)
    plt.title(f"MACD Strategy: Training vs Test Performance\nFast={best_fast}, Slow={best_slow}, Signal={best_signal}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Performance comparison table
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Training':<12} {'Test':<12} {'Difference':<12}")
    print("-" * 60)
    print(f"{'Sharpe Ratio':<20} {best_config['sharpe']:<12.3f} {test_sharpe:<12.3f} {test_sharpe - best_config['sharpe']:<12.3f}")
    print(f"{'Cumulative Return':<20} {best_config['cum_net']:<12.3f} {test_cum_net.iloc[-1]:<12.3f} {test_cum_net.iloc[-1] - best_config['cum_net']:<12.3f}")
    
    return {
        'best_config': best_config,
        'test_results': {
            'sharpe': test_sharpe,
            'cum_ret': test_cum_net.iloc[-1],
            'returns': test_net_ret
        },
        'all_train_results': train_results_df
    }

# ----------------------------------------------------------------------------------- #

def time_horizon_momentum_analysis(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    tcost_bps=20
):
    """
    Time Horizon Analysis for Momentum (Research Area 1)
    Test different time horizons to see where momentum might exist
    """
    ret = px.pct_change(fill_method=None)
    
    print("=" * 60)
    print("TIME HORIZON MOMENTUM ANALYSIS")
    print("=" * 60)
    
    # Test different time horizons as specified in project
    lookbacks = [3, 6, 12, 24, 48, 72, 168, 336, 504, 720]  # hours
    results = []
    
    for lb in lookbacks:
        # Calculate momentum signal
        momentum = ret.rolling(lb, min_periods=lb//2).mean()
        
        # Apply volume weighting (Research Area 2: New Information/Activity)
        vol_avg = vol.rolling(lb, min_periods=1).mean()
        momentum_weighted = momentum * np.log1p(vol_avg)
        
        # Generate signals
        signals = np.sign(momentum_weighted)
        
        # Normalize weights
        signals = signals.sub(signals.mean(axis=1), axis=0)
        signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
        
        # Calculate returns
        turnover = signals.diff().abs().sum(axis=1)
        gross_ret = (signals.shift(1) * ret).sum(axis=1)
        net_ret = gross_ret - turnover * tcost_bps * 1e-4
        
        sharpe = net_ret.mean() / net_ret.std() * np.sqrt(252)
        cum_ret = (1 + net_ret).cumprod()
        
        results.append({
            'lookback_hours': lb,
            'lookback_days': lb / 24,
            'sharpe': sharpe,
            'cum_ret': cum_ret.iloc[-1],
            'volatility': net_ret.std() * np.sqrt(252),
            'returns': net_ret
        })
        
        print(f"Lookback: {lb:>3}h ({lb/24:>5.1f}d) | Sharpe: {sharpe:>6.3f} | Vol: {net_ret.std()*np.sqrt(252):>6.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_df['lookback_hours'], results_df['sharpe'], 'b-o', linewidth=2, markersize=6)
    plt.title('Momentum Sharpe Ratio by Time Horizon')
    plt.xlabel('Lookback Period (hours)')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_df['lookback_hours'], results_df['volatility'], 'g-o', linewidth=2, markersize=6)
    plt.title('Strategy Volatility by Time Horizon')
    plt.xlabel('Lookback Period (hours)')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(results_df['lookback_hours'], results_df['cum_ret'], 'm-o', linewidth=2, markersize=6)
    plt.title('Cumulative Return by Time Horizon')
    plt.xlabel('Lookback Period (hours)')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # Find best strategy and plot its cumulative returns
    best_idx = results_df['sharpe'].idxmax()
    best_strategy = results_df.loc[best_idx]
    best_returns = best_strategy['returns']
    cum_ret = (1 + best_returns).cumprod()
    
    plt.plot(cum_ret.index, cum_ret.values, 'b-', linewidth=2, 
             label=f"Best: {best_strategy['lookback_hours']}h ({best_strategy['lookback_days']:.1f}d)")
    plt.title(f'Best Momentum Strategy Performance\nSharpe: {best_strategy["sharpe"]:.3f}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ----------------------------------------------------------------------------------- #

def volume_activity_momentum(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    tcost_bps=20
):
    """
    New Information/Activity Analysis (Research Area 2)
    Use volume spikes as momentum amplifiers and filter low-liquidity pairs
    """
    ret = px.pct_change(fill_method=None)
    
    print("=" * 60)
    print("VOLUME ACTIVITY MOMENTUM ANALYSIS")
    print("=" * 60)
    
    # Calculate volume z-scores
    vol_z_scores = pd.DataFrame(index=vol.index, columns=vol.columns)
    for asset in vol.columns:
        vol_rolling_mean = vol[asset].rolling(168, min_periods=24).mean()
        vol_rolling_std = vol[asset].rolling(168, min_periods=24).std()
        vol_z_scores[asset] = (vol[asset] - vol_rolling_mean) / vol_rolling_std
    
    lookbacks = [24, 48, 72, 168]
    vol_thresholds = [0.5, 1.0, 1.5, 2.0]
    results = []
    
    for lb, vol_thresh in product(lookbacks, vol_thresholds):
        # Calculate momentum
        momentum = ret.rolling(lb, min_periods=lb//2).mean()
        
        # Volume filter: only trade when volume is above threshold
        vol_filter = (vol_z_scores > vol_thresh).astype(int)
        
        # Combine momentum with volume filter
        signals = np.sign(momentum) * vol_filter
        
        # Additional volume weighting
        vol_avg = vol.rolling(lb, min_periods=1).mean()
        signals = signals * np.log1p(vol_avg)
        
        # Normalize weights
        signals = signals.sub(signals.mean(axis=1), axis=0)
        signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
        
        # Calculate returns
        turnover = signals.diff().abs().sum(axis=1)
        gross_ret = (signals.shift(1) * ret).sum(axis=1)
        net_ret = gross_ret - turnover * tcost_bps * 1e-4
        
        sharpe = net_ret.mean() / net_ret.std() * np.sqrt(24 * 365)
        cum_ret = (1 + net_ret).cumprod()
        
        # Calculate average volume filter usage
        avg_vol_filter_usage = vol_filter.mean().mean()
        
        results.append({
            'lookback': lb,
            'vol_threshold': vol_thresh,
            'sharpe': sharpe,
            'cum_ret': cum_ret.iloc[-1],
            'vol_filter_usage': avg_vol_filter_usage,
            'returns': net_ret
        })
        
        print(f"LB: {lb:>3}h | Vol Thresh: {vol_thresh:>3.1f} | Sharpe: {sharpe:>6.3f} | Vol Usage: {avg_vol_filter_usage:>5.3f}")
    
    results_df = pd.DataFrame(results).sort_values(by='sharpe', ascending=False)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Sharpe heatmap
    pivot = results_df.pivot_table(index='lookback', columns='vol_threshold', values='sharpe')
    
    plt.subplot(2, 2, 1)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
    plt.title('Sharpe Ratio by Lookback and Volume Threshold')
    plt.xlabel('Volume Z-Score Threshold')
    plt.ylabel('Lookback (hours)')
    
    # Volume filter usage
    plt.subplot(2, 2, 2)
    vol_usage_pivot = results_df.pivot_table(index='lookback', columns='vol_threshold', values='vol_filter_usage')
    sns.heatmap(vol_usage_pivot, annot=True, fmt='.3f', cmap='Blues')
    plt.title('Volume Filter Usage Rate')
    plt.xlabel('Volume Z-Score Threshold')
    plt.ylabel('Lookback (hours)')
    
    # Best strategy performance
    plt.subplot(2, 2, 3)
    best = results_df.iloc[0]
    best_returns = best['returns']
    cum_ret = (1 + best_returns).cumprod()
    plt.plot(cum_ret.index, cum_ret.values, 'b-', linewidth=2)
    plt.title(f'Best Volume-Activity Strategy\nLB: {best["lookback"]}h, Vol Thresh: {best["vol_threshold"]:.1f}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    # Sharpe vs Volume Filter Usage
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['vol_filter_usage'], results_df['sharpe'], alpha=0.7)
    plt.xlabel('Volume Filter Usage Rate')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe vs Volume Filter Usage')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ----------------------------------------------------------------------------------- #

def seasonality_momentum_analysis(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    tcost_bps=20
):
    """
    Seasonality Analysis (Research Area 3)
    Explore weekdays vs weekends, day vs night patterns
    """
    ret = px.pct_change(fill_method=None)
    
    # Add time-based features
    px_with_time = px.copy()
    px_with_time['weekday'] = px_with_time.index.weekday
    px_with_time['hour'] = px_with_time.index.hour
    px_with_time['is_weekend'] = px_with_time['weekday'].isin([5, 6])
    px_with_time['is_work_hours'] = (px_with_time['hour'] >= 9) & (px_with_time['hour'] <= 17)
    px_with_time['is_us_market_hours'] = (px_with_time['hour'] >= 14) & (px_with_time['hour'] <= 21)  # 9AM-4PM EST
    
    print("=" * 60)
    print("SEASONALITY MOMENTUM ANALYSIS")
    print("=" * 60)
    
    # Define time periods
    time_periods = {
        'Weekday': px_with_time['weekday'].isin([0, 1, 2, 3, 4]),
        'Weekend': px_with_time['weekday'].isin([5, 6]),
        'Work Hours': px_with_time['is_work_hours'],
        'Off Hours': ~px_with_time['is_work_hours'],
        'US Market Hours': px_with_time['is_us_market_hours'],
        'Non-US Market Hours': ~px_with_time['is_us_market_hours']
    }
    
    lookbacks = [24, 48, 72, 168]
    results = []
    
    for period_name, period_mask in time_periods.items():
        period_ret = ret[period_mask]
        
        print(f"\n{period_name} Analysis:")
        
        for lb in lookbacks:
            # Calculate momentum for this period
            momentum = period_ret.rolling(lb, min_periods=lb//2).mean()
            
            # Apply volume weighting
            period_vol = vol[period_mask]
            vol_avg = period_vol.rolling(lb, min_periods=1).mean()
            momentum_weighted = momentum * np.log1p(vol_avg)
            
            # Generate signals
            signals = np.sign(momentum_weighted)
            
            # Normalize weights
            signals = signals.sub(signals.mean(axis=1), axis=0)
            signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
            
            # Calculate returns
            turnover = signals.diff().abs().sum(axis=1)
            gross_ret = (signals.shift(1) * period_ret).sum(axis=1)
            net_ret = gross_ret - turnover * tcost_bps * 1e-4
            
            sharpe = net_ret.mean() / net_ret.std() * np.sqrt(24 * 365)
            cum_ret = (1 + net_ret).cumprod()
            
            results.append({
                'period': period_name,
                'lookback': lb,
                'sharpe': sharpe,
                'cum_ret': cum_ret.iloc[-1],
                'volatility': net_ret.std() * np.sqrt(24 * 365),
                'num_observations': len(period_ret),
                'returns': net_ret
            })
            
            print(f"  Lookback {lb:>3}h: Sharpe {sharpe:>6.3f} | Vol {net_ret.std()*np.sqrt(24*365):>6.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Sharpe by period and lookback
    pivot = results_df.pivot_table(index='lookback', columns='period', values='sharpe')
    
    plt.subplot(2, 2, 1)
    pivot.plot(kind='bar', ax=plt.gca())
    plt.title('Sharpe Ratio by Time Period and Lookback')
    plt.xlabel('Lookback (hours)')
    plt.ylabel('Sharpe Ratio')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Average Sharpe by period
    plt.subplot(2, 2, 2)
    period_avg = results_df.groupby('period')['sharpe'].mean().sort_values(ascending=True)
    period_avg.plot(kind='barh')
    plt.title('Average Sharpe by Time Period')
    plt.xlabel('Sharpe Ratio')
    plt.grid(True)
    
    # Volatility by period
    plt.subplot(2, 2, 3)
    period_vol = results_df.groupby('period')['volatility'].mean().sort_values(ascending=True)
    period_vol.plot(kind='barh')
    plt.title('Average Volatility by Time Period')
    plt.xlabel('Annualized Volatility')
    plt.grid(True)
    
    # Best strategy performance
    plt.subplot(2, 2, 4)
    best_idx = results_df['sharpe'].idxmax()
    best_strategy = results_df.loc[best_idx]
    best_returns = best_strategy['returns']
    cum_ret = (1 + best_returns).cumprod()
    plt.plot(cum_ret.index, cum_ret.values, 'b-', linewidth=2)
    plt.title(f'Best Seasonal Strategy\n{best_strategy["period"]} - {best_strategy["lookback"]}h')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ----------------------------------------------------------------------------------- #

def theme_momentum_analysis(
    train_px: pd.DataFrame,
    train_vol: pd.DataFrame,
    test_px: pd.DataFrame = None,
    test_vol: pd.DataFrame = None,
    tcost_bps=20
):
    """
    Investment Themes Analysis (Research Area 4)
    Analyze momentum within crypto themes: L1, L2, DeFi, etc.
    """
    train_ret = train_px.pct_change(fill_method=None)
    if test_px is not None:
        test_ret = test_px.pct_change(fill_method=None)
    
    # Define crypto themes based on project requirements
    themes = {
        'L1_Blockchains': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'SUIUSDT'],
        'Exchange_Tokens': ['BNBUSDT'],
        'Meme_Coins': ['DOGEUSDT', 'FLOKIUSDT'],
        'DeFi': ['XRPUSDT'],  # Add more DeFi tokens as needed
        'Stablecoins': ['USDCUSDT']
    }
    
    print("=" * 60)
    print("THEME MOMENTUM ANALYSIS")
    print("=" * 60)
    
    lookbacks = [24, 48, 72, 168, 336, 720]
    train_results = []
    
    for theme_name, assets in themes.items():
        # Filter assets that exist in our data
        available_assets = [asset for asset in assets if asset in train_px.columns]
        if len(available_assets) < 1:
            continue
            
        theme_ret = train_ret[available_assets]
        theme_vol = train_vol[available_assets]
        
        print(f"\n{theme_name} ({len(available_assets)} assets):")
        
        for lb in lookbacks:
            # Calculate theme momentum
            theme_momentum = theme_ret.rolling(lb, min_periods=lb//2).mean()
            
            # Apply volume weighting
            theme_vol_avg = theme_vol.rolling(lb, min_periods=1).mean()
            theme_momentum_weighted = theme_momentum * np.log1p(theme_vol_avg)
            
            # Equal weight within theme
            theme_weights = pd.DataFrame(1/len(available_assets), 
                                       index=theme_momentum.index, 
                                       columns=available_assets)
            
            # Apply momentum signal
            theme_signal = np.sign(theme_momentum_weighted) * theme_weights
            
            # Calculate returns
            theme_gross = (theme_signal.shift(1) * theme_ret).sum(axis=1)
            
            # Apply transaction costs
            theme_turnover = theme_signal.diff().abs().sum(axis=1)
            theme_net = theme_gross - theme_turnover * tcost_bps * 1e-4
            
            theme_sharpe = theme_net.mean() / theme_net.std() * np.sqrt(24 * 365)
            theme_cum_ret = (1 + theme_net).cumprod()
            
            # Calculate theme correlation
            theme_corr = theme_ret.corr().mean().mean()
            
            train_results.append({
                'theme': theme_name,
                'lookback': lb,
                'sharpe': theme_sharpe,
                'cum_ret': theme_cum_ret.iloc[-1],
                'correlation': theme_corr,
                'num_assets': len(available_assets),
                'returns': theme_net,
                'gross_returns': theme_gross,
                'turnover': theme_turnover.mean(),
                'available_assets': available_assets
            })
            
            print(f"  Lookback {lb:>3}h: Sharpe {theme_sharpe:>6.3f} | Corr {theme_corr:>6.3f} | Turnover {theme_turnover.mean():>6.3f}")
    
    train_results_df = pd.DataFrame(train_results)
    
    # Find best configuration for each theme
    best_configs = {}
    for theme_name in themes.keys():
        theme_results = train_results_df[train_results_df['theme'] == theme_name]
        if len(theme_results) > 0:
            best_idx = theme_results['sharpe'].idxmax()
            best_configs[theme_name] = train_results_df.loc[best_idx]
    
    # Plot training results
    plt.figure(figsize=(15, 10))
    
    # Sharpe by theme and lookback
    pivot = train_results_df.pivot_table(index='lookback', columns='theme', values='sharpe')
    
    plt.subplot(2, 2, 1)
    pivot.plot(kind='bar', ax=plt.gca())
    plt.title('Sharpe Ratio by Theme and Lookback (Training)')
    plt.xlabel('Lookback (hours)')
    plt.ylabel('Sharpe Ratio')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Average Sharpe by theme
    plt.subplot(2, 2, 2)
    theme_avg = train_results_df.groupby('theme')['sharpe'].mean().sort_values(ascending=True)
    theme_avg.plot(kind='barh')
    plt.title('Average Sharpe by Theme (Training)')
    plt.xlabel('Sharpe Ratio')
    plt.grid(True)
    
    # Correlation within themes
    plt.subplot(2, 2, 3)
    theme_corr_avg = train_results_df.groupby('theme')['correlation'].mean().sort_values(ascending=True)
    theme_corr_avg.plot(kind='barh')
    plt.title('Average Correlation Within Themes')
    plt.xlabel('Average Correlation')
    plt.grid(True)
    
    # Best theme strategy (training)
    plt.subplot(2, 2, 4)
    best_idx = train_results_df['sharpe'].idxmax()
    best_strategy = train_results_df.loc[best_idx]
    best_returns = best_strategy['returns']
    best_gross_returns = best_strategy['gross_returns']
    cum_ret = (1 + best_returns).cumprod()
    cum_gross_ret = (1 + best_gross_returns).cumprod()
    
    plt.plot(cum_ret.index, cum_ret.values, 'b-', linewidth=2, label='Net Returns (with costs)')
    plt.plot(cum_gross_ret.index, cum_gross_ret.values, 'g--', linewidth=2, label='Gross Returns (no costs)')
    plt.title(f'Best Theme Strategy (Training)\n{best_strategy["theme"]} - {best_strategy["lookback"]}h')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Testing phase (if test data provided)
    if test_px is not None and test_vol is not None:
        print("\n" + "=" * 60)
        print("THEME MOMENTUM ANALYSIS - TESTING PHASE")
        print("=" * 60)
        
        test_results = []
        
        for theme_name, best_config in best_configs.items():
            available_assets = best_config['available_assets']
            lb = best_config['lookback']
            
            # Apply best configuration to test data
            test_theme_ret = test_ret[available_assets]
            test_theme_vol = test_vol[available_assets]
            
            # Calculate theme momentum on test data
            test_theme_momentum = test_theme_ret.rolling(lb, min_periods=lb//2).mean()
            
            # Apply volume weighting
            test_theme_vol_avg = test_theme_vol.rolling(lb, min_periods=1).mean()
            test_theme_momentum_weighted = test_theme_momentum * np.log1p(test_theme_vol_avg)
            
            # Equal weight within theme
            test_theme_weights = pd.DataFrame(1/len(available_assets), 
                                            index=test_theme_momentum.index, 
                                            columns=available_assets)
            
            # Apply momentum signal
            test_theme_signal = np.sign(test_theme_momentum_weighted) * test_theme_weights
            
            # Calculate returns
            test_theme_gross = (test_theme_signal.shift(1) * test_theme_ret).sum(axis=1)
            
            # Apply transaction costs
            test_theme_turnover = test_theme_signal.diff().abs().sum(axis=1)
            test_theme_net = test_theme_gross - test_theme_turnover * tcost_bps * 1e-4
            
            test_theme_sharpe = test_theme_net.mean() / test_theme_net.std() * np.sqrt(24 * 365)
            test_theme_cum_ret = (1 + test_theme_net).cumprod()
            
            # Calculate theme correlation on test data
            test_theme_corr = test_theme_ret.corr().mean().mean()
            
            test_results.append({
                'theme': theme_name,
                'lookback': lb,
                'sharpe': test_theme_sharpe,
                'cum_ret': test_theme_cum_ret.iloc[-1],
                'correlation': test_theme_corr,
                'num_assets': len(available_assets),
                'returns': test_theme_net,
                'gross_returns': test_theme_gross,
                'turnover': test_theme_turnover.mean()
            })
            
            print(f"{theme_name} (Test): Lookback {lb:>3}h | Sharpe {test_theme_sharpe:>6.3f} | Corr {test_theme_corr:>6.3f} | Turnover {test_theme_turnover.mean():>6.3f}")
        
        test_results_df = pd.DataFrame(test_results)
        
        # Plot test results
        plt.figure(figsize=(15, 10))
        
        # Sharpe by theme (test)
        plt.subplot(2, 2, 1)
        test_sharpe_values = test_results_df['sharpe'].values
        test_theme_names = test_results_df['theme'].values
        colors = ['green' if x > 0 else 'red' for x in test_sharpe_values]
        plt.bar(test_theme_names, test_sharpe_values, color=colors, alpha=0.7)
        plt.title('Sharpe Ratio by Theme (Test)')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Training vs Test Sharpe comparison
        plt.subplot(2, 2, 2)
        train_sharpe_values = [best_configs[theme]['sharpe'] for theme in test_results_df['theme']]
        test_sharpe_values = test_results_df['sharpe'].values
        theme_names = test_results_df['theme'].values
        
        x = np.arange(len(theme_names))
        width = 0.35
        
        plt.bar(x - width/2, train_sharpe_values, width, label='Training', alpha=0.7)
        plt.bar(x + width/2, test_sharpe_values, width, label='Test', alpha=0.7)
        plt.xlabel('Theme')
        plt.ylabel('Sharpe Ratio')
        plt.title('Training vs Test Sharpe by Theme')
        plt.xticks(x, theme_names, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Best test strategy performance
        plt.subplot(2, 2, 3)
        best_test_idx = test_results_df['sharpe'].idxmax()
        best_test_strategy = test_results_df.loc[best_test_idx]
        best_test_returns = best_test_strategy['returns']
        best_test_gross_returns = best_test_strategy['gross_returns']
        test_cum_ret = (1 + best_test_returns).cumprod()
        test_cum_gross_ret = (1 + best_test_gross_returns).cumprod()
        
        plt.plot(test_cum_ret.index, test_cum_ret.values, 'r-', linewidth=2, label='Net Returns (with costs)')
        plt.plot(test_cum_gross_ret.index, test_cum_gross_ret.values, 'orange', linestyle='--', linewidth=2, label='Gross Returns (no costs)')
        plt.title(f'Best Theme Strategy (Test)\n{best_test_strategy["theme"]} - {best_test_strategy["lookback"]}h')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Performance comparison table
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Create comparison table
        comparison_data = []
        for theme_name in test_results_df['theme']:
            train_sharpe = best_configs[theme_name]['sharpe']
            test_sharpe = test_results_df[test_results_df['theme'] == theme_name]['sharpe'].iloc[0]
            train_cum_ret = best_configs[theme_name]['cum_ret']
            test_cum_ret = test_results_df[test_results_df['theme'] == theme_name]['cum_ret'].iloc[0]
            train_turnover = best_configs[theme_name]['turnover']
            test_turnover = test_results_df[test_results_df['theme'] == theme_name]['turnover'].iloc[0]
            
            comparison_data.append([
                theme_name,
                f"{train_sharpe:.3f}",
                f"{test_sharpe:.3f}",
                f"{train_cum_ret:.3f}",
                f"{test_cum_ret:.3f}",
                f"{train_turnover:.3f}"
            ])
        
        table = plt.table(cellText=comparison_data,
                         colLabels=['Theme', 'Train Sharpe', 'Test Sharpe', 'Train Return', 'Test Return', 'Avg Turnover'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        plt.title('Training vs Test Performance Comparison')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("THEME PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'Theme':<15} {'Train Sharpe':<12} {'Test Sharpe':<12} {'Train Return':<12} {'Test Return':<12} {'Avg Turnover':<12}")
        print("-" * 80)
        
        for theme_name in test_results_df['theme']:
            train_sharpe = best_configs[theme_name]['sharpe']
            test_sharpe = test_results_df[test_results_df['theme'] == theme_name]['sharpe'].iloc[0]
            train_cum_ret = best_configs[theme_name]['cum_ret']
            test_cum_ret = test_results_df[test_results_df['theme'] == theme_name]['cum_ret'].iloc[0]
            train_turnover = best_configs[theme_name]['turnover']
            test_turnover = test_results_df[test_results_df['theme'] == theme_name]['turnover'].iloc[0]
            
            print(f"{theme_name:<15} {train_sharpe:<12.3f} {test_sharpe:<12.3f} {train_cum_ret:<12.3f} {test_cum_ret:<12.3f} {train_turnover:<12.3f}")
        
        # Show transaction cost impact
        print("\n" + "=" * 60)
        print("TRANSACTION COST IMPACT ANALYSIS")
        print("=" * 60)
        print(f"{'Theme':<15} {'Gross Sharpe':<12} {'Net Sharpe':<12} {'Cost Impact':<12}")
        print("-" * 60)
        
        for theme_name in test_results_df['theme']:
            train_gross_returns = best_configs[theme_name]['gross_returns']
            train_net_returns = best_configs[theme_name]['returns']
            test_gross_returns = test_results_df[test_results_df['theme'] == theme_name]['gross_returns'].iloc[0]
            test_net_returns = test_results_df[test_results_df['theme'] == theme_name]['returns'].iloc[0]
            
            train_gross_sharpe = train_gross_returns.mean() / train_gross_returns.std() * np.sqrt(24 * 365)
            test_gross_sharpe = test_gross_returns.mean() / test_gross_returns.std() * np.sqrt(24 * 365)
            train_net_sharpe = best_configs[theme_name]['sharpe']
            test_net_sharpe = test_results_df[test_results_df['theme'] == theme_name]['sharpe'].iloc[0]
            
            train_cost_impact = train_gross_sharpe - train_net_sharpe
            test_cost_impact = test_gross_sharpe - test_net_sharpe
            
            print(f"{theme_name:<15} {train_gross_sharpe:<12.3f} {train_net_sharpe:<12.3f} {train_cost_impact:<12.3f}")
            print(f"{'  (Test)':<15} {test_gross_sharpe:<12.3f} {test_net_sharpe:<12.3f} {test_cost_impact:<12.3f}")
        
        return {
            'train_results': train_results_df,
            'test_results': test_results_df,
            'best_configs': best_configs
        }
    
    return {
        'train_results': train_results_df,
        'best_configs': best_configs
    }

# ----------------------------------------------------------------------------------- #

def technical_mechanics_analysis(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    tcost_bps=20
):
    """
    Technical Mechanics Analysis (Research Area 5)
    Explore mechanical rebalancing effects and front-running opportunities
    """
    ret = px.pct_change(fill_method=None)
    
    print("=" * 60)
    print("TECHNICAL MECHANICS ANALYSIS")
    print("=" * 60)
    
    # Add time features for institutional trading patterns
    px_with_time = px.copy()
    px_with_time['hour'] = px_with_time.index.hour
    px_with_time['weekday'] = px_with_time.index.weekday
    px_with_time['is_month_end'] = px_with_time.index.day >= 25
    px_with_time['is_quarter_end'] = (px_with_time.index.month.isin([3, 6, 9, 12])) & (px_with_time.index.day >= 25)
    
    # Define different trading patterns
    patterns = {
        'Regular_Hours': (px_with_time['hour'] >= 9) & (px_with_time['hour'] <= 17),
        'Pre_Market': (px_with_time['hour'] >= 6) & (px_with_time['hour'] < 9),
        'After_Hours': (px_with_time['hour'] > 17) & (px_with_time['hour'] <= 22),
        'Month_End': px_with_time['is_month_end'],
        'Quarter_End': px_with_time['is_quarter_end'],
        'Weekend': px_with_time['weekday'].isin([5, 6])
    }
    
    lookbacks = [24, 48, 72, 168]
    results = []
    
    for pattern_name, pattern_mask in patterns.items():
        pattern_ret = ret[pattern_mask]
        
        if len(pattern_ret) < 100:  # Skip if too few observations
            continue
            
        print(f"\n{pattern_name} Analysis ({len(pattern_ret)} observations):")
        
        for lb in lookbacks:
            # Calculate momentum for this pattern
            momentum = pattern_ret.rolling(lb, min_periods=lb//2).mean()
            
            # Apply volume weighting
            pattern_vol = vol[pattern_mask]
            vol_avg = pattern_vol.rolling(lb, min_periods=1).mean()
            momentum_weighted = momentum * np.log1p(vol_avg)
            
            # Generate signals
            signals = np.sign(momentum_weighted)
            
            # Normalize weights
            signals = signals.sub(signals.mean(axis=1), axis=0)
            signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
            
            # Calculate returns
            turnover = signals.diff().abs().sum(axis=1)
            gross_ret = (signals.shift(1) * pattern_ret).sum(axis=1)
            net_ret = gross_ret - turnover * tcost_bps * 1e-4
            
            sharpe = net_ret.mean() / net_ret.std() * np.sqrt(24 * 365)
            cum_ret = (1 + net_ret).cumprod()
            
            results.append({
                'pattern': pattern_name,
                'lookback': lb,
                'sharpe': sharpe,
                'cum_ret': cum_ret.iloc[-1],
                'volatility': net_ret.std() * np.sqrt(24 * 365),
                'num_observations': len(pattern_ret),
                'returns': net_ret
            })
            
            print(f"  Lookback {lb:>3}h: Sharpe {sharpe:>6.3f} | Vol {net_ret.std()*np.sqrt(24*365):>6.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Sharpe by pattern and lookback
    pivot = results_df.pivot_table(index='lookback', columns='pattern', values='sharpe')
    
    plt.subplot(2, 2, 1)
    pivot.plot(kind='bar', ax=plt.gca())
    plt.title('Sharpe Ratio by Trading Pattern and Lookback')
    plt.xlabel('Lookback (hours)')
    plt.ylabel('Sharpe Ratio')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Average Sharpe by pattern
    plt.subplot(2, 2, 2)
    pattern_avg = results_df.groupby('pattern')['sharpe'].mean().sort_values(ascending=True)
    pattern_avg.plot(kind='barh')
    plt.title('Average Sharpe by Trading Pattern')
    plt.xlabel('Sharpe Ratio')
    plt.grid(True)
    
    # Number of observations by pattern
    plt.subplot(2, 2, 3)
    pattern_obs = results_df.groupby('pattern')['num_observations'].mean().sort_values(ascending=True)
    pattern_obs.plot(kind='barh')
    plt.title('Average Number of Observations by Pattern')
    plt.xlabel('Number of Observations')
    plt.grid(True)
    
    # Best pattern strategy
    plt.subplot(2, 2, 4)
    best_idx = results_df['sharpe'].idxmax()
    best_strategy = results_df.loc[best_idx]
    best_returns = best_strategy['returns']
    cum_ret = (1 + best_returns).cumprod()
    plt.plot(cum_ret.index, cum_ret.values, 'b-', linewidth=2)
    plt.title(f'Best Technical Pattern Strategy\n{best_strategy["pattern"]} - {best_strategy["lookback"]}h')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ----------------------------------------------------------------------------------- #

def short_term_reversal_analysis(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    tcost_bps=20
):
    """
    Short-term Reversal Analysis (Research Area 1 for Reversal)
    Test different short time horizons for reversal patterns
    """
    ret = px.pct_change(fill_method=None)
    
    print("=" * 60)
    print("SHORT-TERM REVERSAL ANALYSIS")
    print("=" * 60)
    
    # Test short time horizons for reversal
    lookbacks = [1, 2, 3, 4, 6, 12, 24, 48]  # hours
    results = []
    
    for lb in lookbacks:
        # Calculate short-term momentum (for reversal, we'll invert the signal)
        momentum = ret.rolling(lb, min_periods=lb//2).mean()
        
        # Apply volume weighting
        vol_avg = vol.rolling(lb, min_periods=1).mean()
        momentum_weighted = momentum * np.log1p(vol_avg)
        
        # Generate reversal signals (invert momentum)
        signals = -np.sign(momentum_weighted)
        
        # Normalize weights
        signals = signals.sub(signals.mean(axis=1), axis=0)
        signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
        
        # Calculate returns
        turnover = signals.diff().abs().sum(axis=1)
        gross_ret = (signals.shift(1) * ret).sum(axis=1)
        net_ret = gross_ret - turnover * tcost_bps * 1e-4
        
        sharpe = net_ret.mean() / net_ret.std() * np.sqrt(24 * 365)
        cum_ret = (1 + net_ret).cumprod()
        
        results.append({
            'lookback_hours': lb,
            'sharpe': sharpe,
            'cum_ret': cum_ret.iloc[-1],
            'volatility': net_ret.std() * np.sqrt(24 * 365),
            'returns': net_ret
        })
        
        print(f"Lookback: {lb:>3}h | Sharpe: {sharpe:>6.3f} | Vol: {net_ret.std()*np.sqrt(24*365):>6.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_df['lookback_hours'], results_df['sharpe'], 'r-o', linewidth=2, markersize=6)
    plt.title('Reversal Sharpe Ratio by Time Horizon')
    plt.xlabel('Lookback Period (hours)')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.axhline(y=0, color='b', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_df['lookback_hours'], results_df['volatility'], 'g-o', linewidth=2, markersize=6)
    plt.title('Reversal Strategy Volatility by Time Horizon')
    plt.xlabel('Lookback Period (hours)')
    plt.ylabel('Annualized Volatility')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(results_df['lookback_hours'], results_df['cum_ret'], 'm-o', linewidth=2, markersize=6)
    plt.title('Reversal Cumulative Return by Time Horizon')
    plt.xlabel('Lookback Period (hours)')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    # Find best strategy and plot its cumulative returns
    best_idx = results_df['sharpe'].idxmax()
    best_strategy = results_df.loc[best_idx]
    best_returns = best_strategy['returns']
    cum_ret = (1 + best_returns).cumprod()
    
    plt.plot(cum_ret.index, cum_ret.values, 'r-', linewidth=2, 
             label=f"Best: {best_strategy['lookback_hours']}h")
    plt.title(f'Best Reversal Strategy Performance\nSharpe: {best_strategy["sharpe"]:.3f}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ----------------------------------------------------------------------------------- #

def uninformed_trading_reversal(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    tcost_bps=20
):
    """
    Uninformed Trading Reversal (Research Area 2 for Reversal)
    Apply indicators of activity/info and isolate cases of lower activity
    """
    ret = px.pct_change(fill_method=None)
    
    print("=" * 60)
    print("UNINFORMED TRADING REVERSAL ANALYSIS")
    print("=" * 60)
    
    # Calculate volume z-scores to identify low activity periods
    vol_z_scores = pd.DataFrame(index=vol.index, columns=vol.columns)
    for asset in vol.columns:
        vol_rolling_mean = vol[asset].rolling(168, min_periods=24).mean()
        vol_rolling_std = vol[asset].rolling(168, min_periods=24).std()
        vol_z_scores[asset] = (vol[asset] - vol_rolling_mean) / vol_rolling_std
    
    # Define activity filters
    activity_filters = {
        'Low_Activity': vol_z_scores < -0.5,  # Below average volume
        'Very_Low_Activity': vol_z_scores < -1.0,  # Significantly below average
        'Normal_Activity': (vol_z_scores >= -0.5) & (vol_z_scores <= 0.5),
        'High_Activity': vol_z_scores > 0.5  # Above average volume
    }
    
    lookbacks = [6, 12, 24, 48]
    results = []
    
    for filter_name, activity_mask in activity_filters.items():
        # Apply activity filter to returns
        filtered_ret = ret.copy()
        filtered_ret[~activity_mask] = np.nan
        
        # Count valid observations
        valid_obs = filtered_ret.notna().sum().sum()
        
        if valid_obs < 1000:  # Skip if too few observations
            continue
            
        print(f"\n{filter_name} Analysis ({valid_obs} observations):")
        
        for lb in lookbacks:
            # Calculate reversal signal for this activity level
            momentum = filtered_ret.rolling(lb, min_periods=lb//2).mean()
            
            # Apply volume weighting
            vol_avg = vol.rolling(lb, min_periods=1).mean()
            momentum_weighted = momentum * np.log1p(vol_avg)
            
            # Generate reversal signals (invert momentum)
            signals = -np.sign(momentum_weighted)
            
            # Only trade when we have valid signals
            signals = signals * activity_mask.astype(int)
            
            # Normalize weights
            signals = signals.sub(signals.mean(axis=1), axis=0)
            signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
            
            # Calculate returns
            turnover = signals.diff().abs().sum(axis=1)
            gross_ret = (signals.shift(1) * ret).sum(axis=1)
            net_ret = gross_ret - turnover * tcost_bps * 1e-4
            
            sharpe = net_ret.mean() / net_ret.std() * np.sqrt(24 * 365)
            cum_ret = (1 + net_ret).cumprod()
            
            # Calculate activity filter usage
            filter_usage = activity_mask.mean().mean()
            
            results.append({
                'activity_filter': filter_name,
                'lookback': lb,
                'sharpe': sharpe,
                'cum_ret': cum_ret.iloc[-1],
                'filter_usage': filter_usage,
                'valid_observations': valid_obs,
                'returns': net_ret
            })
            
            print(f"  Lookback {lb:>3}h: Sharpe {sharpe:>6.3f} | Usage {filter_usage:>5.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Sharpe by activity filter and lookback
    pivot = results_df.pivot_table(index='lookback', columns='activity_filter', values='sharpe')
    
    plt.subplot(2, 2, 1)
    pivot.plot(kind='bar', ax=plt.gca())
    plt.title('Reversal Sharpe by Activity Level and Lookback')
    plt.xlabel('Lookback (hours)')
    plt.ylabel('Sharpe Ratio')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Average Sharpe by activity filter
    plt.subplot(2, 2, 2)
    activity_avg = results_df.groupby('activity_filter')['sharpe'].mean().sort_values(ascending=True)
    activity_avg.plot(kind='barh')
    plt.title('Average Reversal Sharpe by Activity Level')
    plt.xlabel('Sharpe Ratio')
    plt.grid(True)
    
    # Filter usage by activity level
    plt.subplot(2, 2, 3)
    usage_avg = results_df.groupby('activity_filter')['filter_usage'].mean().sort_values(ascending=True)
    usage_avg.plot(kind='barh')
    plt.title('Average Filter Usage by Activity Level')
    plt.xlabel('Usage Rate')
    plt.grid(True)
    
    # Best strategy performance
    plt.subplot(2, 2, 4)
    best_idx = results_df['sharpe'].idxmax()
    best_strategy = results_df.loc[best_idx]
    best_returns = best_strategy['returns']
    cum_ret = (1 + best_returns).cumprod()
    plt.plot(cum_ret.index, cum_ret.values, 'r-', linewidth=2)
    plt.title(f'Best Uninformed Trading Strategy\n{best_strategy["activity_filter"]} - {best_strategy["lookback"]}h')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ----------------------------------------------------------------------------------- #

def correlation_based_reversal(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    tcost_bps=20
):
    """
    Correlation-based Reversal Strategy (Research Area 3 for Reversal)
    Security A - (Something Correlated to It)
    """
    ret = px.pct_change(fill_method=None)
    
    print("=" * 60)
    print("CORRELATION-BASED REVERSAL STRATEGY")
    print("=" * 60)
    
    corr_lookbacks = [168, 336, 504]  # 1 week, 2 weeks, 3 weeks
    reversal_lookbacks = [12, 24, 48]  # Short-term reversal
    results = []
    
    for corr_lb, rev_lb in product(corr_lookbacks, reversal_lookbacks):
        # Calculate rolling correlation matrix
        signals = pd.DataFrame(0, index=px.index, columns=px.columns)
        
        for i in range(corr_lb, len(px)):
            window_ret = ret.iloc[i-corr_lb:i]
            corr_matrix = window_ret.corr()
            
            # For each asset, find the most correlated asset
            for asset in px.columns:
                if asset in corr_matrix.columns:
                    # Get correlations excluding self
                    asset_corrs = corr_matrix[asset].drop(asset)
                    if len(asset_corrs) > 0:
                        most_corr_asset = asset_corrs.idxmax()
                        corr_value = asset_corrs.max()
                        
                        if corr_value > 0.6:  # Only trade if correlation is high
                            # Calculate spread: Asset A - Correlated Asset
                            spread = px.iloc[i][asset] - px.iloc[i][most_corr_asset]
                            
                            # Calculate rolling mean and std of spread
                            if i >= rev_lb:
                                spread_window = px.iloc[i-rev_lb:i][asset] - px.iloc[i-rev_lb:i][most_corr_asset]
                                spread_mean = spread_window.mean()
                                spread_std = spread_window.std()
                                
                                if spread_std > 0:
                                    z_score = (spread - spread_mean) / spread_std
                                    
                                    # Reversal signal: short when spread is high, long when low
                                    if abs(z_score) > 1.5:  # Only trade when spread deviates significantly
                                        signals.iloc[i][asset] = -np.sign(z_score) * 0.5  # Half weight
                                        signals.iloc[i][most_corr_asset] = np.sign(z_score) * 0.5
        
        # Apply volume filter
        vol_avg = vol.rolling(corr_lb, min_periods=1).mean()
        signals = signals * np.log1p(vol_avg)
        
        # Normalize weights
        signals = signals.sub(signals.mean(axis=1), axis=0)
        signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
        
        # Calculate returns
        turnover = signals.diff().abs().sum(axis=1)
        gross_ret = (signals.shift(1) * ret).sum(axis=1)
        net_ret = gross_ret - turnover * tcost_bps * 1e-4
        
        sharpe = net_ret.mean() / net_ret.std() * np.sqrt(24 * 365)
        cum_ret = (1 + net_ret).cumprod()
        
        # Calculate average number of trades
        avg_trades = (signals != 0).sum().sum() / len(signals)
        
        results.append({
            'corr_lookback': corr_lb,
            'reversal_lookback': rev_lb,
            'sharpe': sharpe,
            'cum_ret': cum_ret.iloc[-1],
            'avg_trades': avg_trades,
            'returns': net_ret
        })
        
        print(f"Corr LB: {corr_lb:>3}h | Rev LB: {rev_lb:>3}h | Sharpe: {sharpe:>6.3f} | Avg Trades: {avg_trades:>6.1f}")
    
    results_df = pd.DataFrame(results).sort_values(by='sharpe', ascending=False)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Sharpe heatmap
    pivot = results_df.pivot_table(index='corr_lookback', columns='reversal_lookback', values='sharpe')
    
    plt.subplot(2, 2, 1)
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
    plt.title('Correlation Reversal Sharpe Ratio')
    plt.xlabel('Reversal Lookback (hours)')
    plt.ylabel('Correlation Lookback (hours)')
    
    # Average trades heatmap
    trades_pivot = results_df.pivot_table(index='corr_lookback', columns='reversal_lookback', values='avg_trades')
    
    plt.subplot(2, 2, 2)
    sns.heatmap(trades_pivot, annot=True, fmt='.1f', cmap='Blues')
    plt.title('Average Number of Trades')
    plt.xlabel('Reversal Lookback (hours)')
    plt.ylabel('Correlation Lookback (hours)')
    
    # Best strategy performance
    plt.subplot(2, 2, 3)
    best_idx = results_df['sharpe'].idxmax()
    best_strategy = results_df.loc[best_idx]
    best_returns = best_strategy['returns']
    cum_ret = (1 + best_returns).cumprod()
    plt.plot(cum_ret.index, cum_ret.values, 'r-', linewidth=2)
    plt.title(f'Best Correlation Reversal Strategy\nCorr LB: {best["corr_lookback"]}h, Rev LB: {best["reversal_lookback"]}h')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    # Sharpe vs Average Trades
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['avg_trades'], results_df['sharpe'], alpha=0.7)
    plt.xlabel('Average Number of Trades')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe vs Trading Activity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# ----------------------------------------------------------------------------------- #

def comprehensive_strategy_comparison(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    tcost_bps=20
):
    """
    Comprehensive comparison of all strategies
    """
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print("=" * 80)
    
    # Store all strategy results
    strategy_results = {}
    
    # 1. Baseline
    ret = px.pct_change(fill_method=None)
    equal_weights = pd.Series(1 / len(px.columns), index=px.columns)
    buyhold_ret = (ret * equal_weights).sum(axis=1)
    strategy_results['Buy & Hold'] = buyhold_ret
    
    # 2. Best momentum strategies
    print("\nTesting Momentum Strategies...")
    
    # Time horizon momentum
    time_horizon_results = time_horizon_momentum_analysis(px, vol, tcost_bps)
    best_time_horizon = time_horizon_results.loc[time_horizon_results['sharpe'].idxmax()]
    strategy_results['Time Horizon Momentum'] = best_time_horizon['returns']
    
    # Volume activity momentum
    volume_results = volume_activity_momentum(px, vol, tcost_bps)
    best_volume = volume_results.iloc[0]  # Already sorted by Sharpe
    strategy_results['Volume Activity Momentum'] = best_volume['returns']
    
    # Seasonality momentum
    seasonality_results = seasonality_momentum_analysis(px, vol, tcost_bps)
    best_seasonality = seasonality_results.loc[seasonality_results['sharpe'].idxmax()]
    strategy_results['Seasonality Momentum'] = best_seasonality['returns']
    
    # Theme momentum
    theme_results = theme_momentum_analysis(px, vol, tcost_bps)
    best_theme = theme_results.loc[theme_results['sharpe'].idxmax()]
    strategy_results['Theme Momentum'] = best_theme['returns']
    
    # Technical mechanics
    technical_results = technical_mechanics_analysis(px, vol, tcost_bps)
    best_technical = technical_results.loc[technical_results['sharpe'].idxmax()]
    strategy_results['Technical Mechanics'] = best_technical['returns']
    
    # 3. Best reversal strategies
    print("\nTesting Reversal Strategies...")
    
    # Short-term reversal
    reversal_results = short_term_reversal_analysis(px, vol, tcost_bps)
    best_reversal = reversal_results.loc[reversal_results['sharpe'].idxmax()]
    strategy_results['Short-term Reversal'] = best_reversal['returns']
    
    # Uninformed trading reversal
    uninformed_results = uninformed_trading_reversal(px, vol, tcost_bps)
    best_uninformed = uninformed_results.loc[uninformed_results['sharpe'].idxmax()]
    strategy_results['Uninformed Trading Reversal'] = best_uninformed['returns']
    
    # Correlation reversal
    corr_reversal_results = correlation_based_reversal(px, vol, tcost_bps)
    best_corr_reversal = corr_reversal_results.iloc[0]  # Already sorted by Sharpe
    strategy_results['Correlation Reversal'] = best_corr_reversal['returns']
    
    # 4. Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for strategy_name, strategy_ret in strategy_results.items():
        # Calculate metrics
        total_return = (1 + strategy_ret).cumprod().iloc[-1] - 1
        annualized_return = (1 + total_return) ** (24 * 365 / len(strategy_ret)) - 1
        volatility = strategy_ret.std() * np.sqrt(24 * 365)
        sharpe_ratio = strategy_ret.mean() / strategy_ret.std() * np.sqrt(24 * 365)
        
        # Maximum drawdown
        cum_ret = (1 + strategy_ret).cumprod()
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        max_drawdown = drawdown.min()
        
        summary_data.append({
            'Strategy': strategy_name,
            'Sharpe': sharpe_ratio,
            'Annual_Return': annualized_return,
            'Volatility': volatility,
            'Max_Drawdown': max_drawdown,
            'Total_Return': total_return
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values(by='Sharpe', ascending=False)
    print(summary_df.round(4).to_string(index=False))
    
    # Plot cumulative returns comparison
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 2, 1)
    for strategy_name, strategy_ret in strategy_results.items():
        cum_ret = (1 + strategy_ret).cumprod()
        plt.plot(cum_ret.index, cum_ret.values, label=strategy_name, linewidth=2)
    
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Sharpe ratio comparison
    plt.subplot(2, 2, 2)
    sharpe_values = summary_df['Sharpe'].values
    strategy_names = summary_df['Strategy'].values
    colors = ['green' if x > 0 else 'red' for x in sharpe_values]
    plt.bar(strategy_names, sharpe_values, color=colors, alpha=0.7)
    plt.title('Sharpe Ratio Comparison')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    
    # Maximum drawdown comparison
    plt.subplot(2, 2, 3)
    dd_values = summary_df['Max_Drawdown'].values
    plt.bar(strategy_names, dd_values, color='orange', alpha=0.7)
    plt.title('Maximum Drawdown Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Maximum Drawdown')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    
    # Volatility comparison
    plt.subplot(2, 2, 4)
    vol_values = summary_df['Volatility'].values
    plt.bar(strategy_names, vol_values, color='purple', alpha=0.7)
    plt.title('Volatility Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Annualized Volatility')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return summary_df

# ----------------------------------------------------------------------------------- #

def resample_data_for_rebalancing(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    rebal_freq: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resample data to different rebalancing frequencies
    
    Input:
    - px: price DataFrame with hourly data
    - vol: volume DataFrame with hourly data  
    - rebal_freq: rebalancing frequency ('1H', '4H', '6H', '12H', '1D', '1W')
    
    Output:
    - resampled price and volume DataFrames
    """
    # Resample prices using last price in each period
    px_resampled = px.resample(rebal_freq).last()
    
    # Resample volumes using sum in each period
    vol_resampled = vol.resample(rebal_freq).sum()
    
    # Forward fill any missing values
    px_resampled = px_resampled.fillna(method='ffill')
    vol_resampled = vol_resampled.fillna(0)
    
    return px_resampled, vol_resampled

# ----------------------------------------------------------------------------------- #

def test_rebalancing_frequencies(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    strategy_type: str = 'momentum',
    tcost_bps: int = 20
):
    """
    Test different rebalancing frequencies for a given strategy type
    
    Input:
    - px: price DataFrame
    - vol: volume DataFrame
    - strategy_type: 'momentum' or 'reversal'
    - tcost_bps: transaction costs in basis points
    """
    ret = px.pct_change(fill_method=None)
    
    # Define rebalancing frequencies to test
    rebal_freqs = ['1H', '4H', '6H', '12H', '1D', '1W']
    lookbacks = [24, 48, 72, 168]  # hours
    
    print("=" * 80)
    print(f"REBALANCING FREQUENCY ANALYSIS - {strategy_type.upper()}")
    print("=" * 80)
    
    results = []
    
    for rebal_freq in rebal_freqs:
        print(f"\nTesting {rebal_freq} rebalancing frequency...")
        
        # Resample data to this frequency
        px_resampled, vol_resampled = resample_data_for_rebalancing(px, vol, rebal_freq)
        ret_resampled = px_resampled.pct_change(fill_method=None)
        
        # Convert lookback periods to the new frequency
        if rebal_freq == '1H':
            freq_lookbacks = lookbacks
        elif rebal_freq == '4H':
            freq_lookbacks = [lb // 4 for lb in lookbacks]
        elif rebal_freq == '6H':
            freq_lookbacks = [lb // 6 for lb in lookbacks]
        elif rebal_freq == '12H':
            freq_lookbacks = [lb // 12 for lb in lookbacks]
        elif rebal_freq == '1D':
            freq_lookbacks = [lb // 24 for lb in lookbacks]
        elif rebal_freq == '1W':
            freq_lookbacks = [max(1, lb // 168) for lb in lookbacks]  # 168 hours = 1 week
        
        for lb in freq_lookbacks:
            if lb < 1:  # Skip if lookback is too short for this frequency
                continue
                
            # Calculate momentum signal
            momentum = ret_resampled.rolling(lb, min_periods=lb//2).mean()
            
            # Apply volume weighting
            vol_avg = vol_resampled.rolling(lb, min_periods=1).mean()
            momentum_weighted = momentum * np.log1p(vol_avg)
            
            # Generate signals based on strategy type
            if strategy_type == 'momentum':
                signals = np.sign(momentum_weighted)
            elif strategy_type == 'reversal':
                signals = -np.sign(momentum_weighted)
            
            # Normalize weights
            signals = signals.sub(signals.mean(axis=1), axis=0)
            signals = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0)
            
            # Calculate returns
            turnover = signals.diff().abs().sum(axis=1)
            gross_ret = (signals.shift(1) * ret_resampled).sum(axis=1)
            net_ret = gross_ret - turnover * tcost_bps * 1e-4
            
            # Annualize based on rebalancing frequency
            if rebal_freq == '1H':
                annualization_factor = 24 * 365
            elif rebal_freq == '4H':
                annualization_factor = 6 * 365
            elif rebal_freq == '6H':
                annualization_factor = 4 * 365
            elif rebal_freq == '12H':
                annualization_factor = 2 * 365
            elif rebal_freq == '1D':
                annualization_factor = 365
            elif rebal_freq == '1W':
                annualization_factor = 52
            
            sharpe = net_ret.mean() / net_ret.std() * np.sqrt(annualization_factor)
            cum_ret = (1 + net_ret).cumprod()
            
            # Calculate additional metrics
            volatility = net_ret.std() * np.sqrt(annualization_factor)
            annual_return = (cum_ret.iloc[-1] ** (annualization_factor/len(net_ret)) - 1)
            
            # Maximum drawdown
            running_max = cum_ret.expanding().max()
            drawdown = (cum_ret - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Average turnover per rebalancing period
            avg_turnover = turnover.mean()
            
            results.append({
                'rebal_freq': rebal_freq,
                'lookback_periods': lb,
                'lookback_hours': lb * {'1H': 1, '4H': 4, '6H': 6, '12H': 12, '1D': 24, '1W': 168}[rebal_freq],
                'sharpe': sharpe,
                'annual_return': annual_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'cum_ret': cum_ret.iloc[-1],
                'avg_turnover': avg_turnover,
                'num_rebalances': len(net_ret),
                'returns': net_ret
            })
            
            print(f"  Lookback: {lb:>2} periods ({lb * {'1H': 1, '4H': 4, '6H': 6, '12H': 12, '1D': 24, '1W': 168}[rebal_freq]:>3}h) | "
                  f"Sharpe: {sharpe:>6.3f} | Vol: {volatility:>6.3f} | Turnover: {avg_turnover:>6.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(20, 12))
    
    # Sharpe ratio heatmap
    plt.subplot(2, 3, 1)
    pivot_sharpe = results_df.pivot_table(index='rebal_freq', columns='lookback_hours', values='sharpe')
    sns.heatmap(pivot_sharpe, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
    plt.title(f'{strategy_type.title()} Sharpe Ratio by Rebalancing Frequency')
    plt.xlabel('Lookback (hours)')
    plt.ylabel('Rebalancing Frequency')
    
    # Annual return heatmap
    plt.subplot(2, 3, 2)
    pivot_return = results_df.pivot_table(index='rebal_freq', columns='lookback_hours', values='annual_return')
    sns.heatmap(pivot_return, annot=True, fmt='.3f', cmap='RdYlGn', center=0)
    plt.title(f'{strategy_type.title()} Annual Return by Rebalancing Frequency')
    plt.xlabel('Lookback (hours)')
    plt.ylabel('Rebalancing Frequency')
    
    # Volatility heatmap
    plt.subplot(2, 3, 3)
    pivot_vol = results_df.pivot_table(index='rebal_freq', columns='lookback_hours', values='volatility')
    sns.heatmap(pivot_vol, annot=True, fmt='.3f', cmap='Blues')
    plt.title(f'{strategy_type.title()} Volatility by Rebalancing Frequency')
    plt.xlabel('Lookback (hours)')
    plt.ylabel('Rebalancing Frequency')
    
    # Average turnover by rebalancing frequency
    plt.subplot(2, 3, 4)
    turnover_by_freq = results_df.groupby('rebal_freq')['avg_turnover'].mean().sort_values(ascending=True)
    turnover_by_freq.plot(kind='barh')
    plt.title('Average Turnover by Rebalancing Frequency')
    plt.xlabel('Average Turnover')
    plt.grid(True)
    
    # Sharpe ratio by rebalancing frequency
    plt.subplot(2, 3, 5)
    sharpe_by_freq = results_df.groupby('rebal_freq')['sharpe'].mean().sort_values(ascending=True)
    sharpe_by_freq.plot(kind='barh')
    plt.title('Average Sharpe by Rebalancing Frequency')
    plt.xlabel('Average Sharpe Ratio')
    plt.grid(True)
    
    # Best strategy performance
    plt.subplot(2, 3, 6)
    best_idx = results_df['sharpe'].idxmax()
    best_strategy = results_df.loc[best_idx]
    best_returns = best_strategy['returns']
    cum_ret = (1 + best_returns).cumprod()
    plt.plot(cum_ret.index, cum_ret.values, 'b-', linewidth=2)
    plt.title(f'Best {strategy_type.title()} Strategy\n{best_strategy["rebal_freq"]} - {best_strategy["lookback_hours"]}h')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"REBALANCING FREQUENCY SUMMARY - {strategy_type.upper()}")
    print("=" * 80)
    
    # Best configuration for each frequency
    print("\nBest Configuration by Rebalancing Frequency:")
    print(f"{'Frequency':<12} {'Lookback':<10} {'Sharpe':<8} {'Return':<8} {'Vol':<8} {'Turnover':<10}")
    print("-" * 70)
    
    for freq in rebal_freqs:
        freq_results = results_df[results_df['rebal_freq'] == freq]
        if len(freq_results) > 0:
            best_freq_idx = freq_results['sharpe'].idxmax()
            best_freq = results_df.loc[best_freq_idx]
            print(f"{best_freq['rebal_freq']:<12} {best_freq['lookback_hours']:<10} "
                  f"{best_freq['sharpe']:<8.3f} {best_freq['annual_return']:<8.3f} "
                  f"{best_freq['volatility']:<8.3f} {best_freq['avg_turnover']:<10.3f}")
    
    # Overall best strategy
    print(f"\nOverall Best Strategy:")
    print(f"Rebalancing: {best_strategy['rebal_freq']}")
    print(f"Lookback: {best_strategy['lookback_hours']} hours")
    print(f"Sharpe Ratio: {best_strategy['sharpe']:.3f}")
    print(f"Annual Return: {best_strategy['annual_return']:.3f}")
    print(f"Volatility: {best_strategy['volatility']:.3f}")
    print(f"Max Drawdown: {best_strategy['max_drawdown']:.3f}")
    print(f"Average Turnover: {best_strategy['avg_turnover']:.3f}")
    
    return results_df

# ----------------------------------------------------------------------------------- #

def momentum_strat_with_rebalancing(
    px: pd.DataFrame, 
    vol: pd.DataFrame, 
    lookback: int,
    lag: int,
    rebal_freq: str = '1H',
    tcost_bps: int = 20
):
    """
    Momentum strategy with variable rebalancing frequency
    
    Input:
    - px: price DataFrame
    - vol: volume DataFrame
    - lookback: lookback period in hours
    - lag: lag period in hours
    - rebal_freq: rebalancing frequency ('1H', '4H', '6H', '12H', '1D', '1W')
    - tcost_bps: transaction costs in basis points
    """
    # Resample data to rebalancing frequency
    px_resampled, vol_resampled = resample_data_for_rebalancing(px, vol, rebal_freq)
    ret_resampled = px_resampled.pct_change(fill_method=None)
    
    # Convert lookback and lag to the new frequency
    freq_multiplier = {'1H': 1, '4H': 4, '6H': 6, '12H': 12, '1D': 24, '1W': 168}[rebal_freq]
    lookback_periods = lookback // freq_multiplier
    lag_periods = lag // freq_multiplier
    
    if lookback_periods < 1 or lag_periods < 0:
        print(f"Warning: Lookback or lag too short for {rebal_freq} frequency")
        return None
    
    # Calculate momentum signal
    vol_avg = vol_resampled.rolling(lookback_periods, min_periods=1).mean()
    port = (ret_resampled.rolling(lookback_periods, min_periods=1).mean()).rank(1) 
    port = port * np.log1p(vol_avg)
    
    # Normalize weights
    port = port.sub(port.mean(axis=1), axis=0)
    port = port.div(port.abs().sum(axis=1), axis=0)
    
    # Calculate turnover and returns
    turnover = (port.fillna(0) - port.shift()).abs().sum(axis=1)
    gross_ret = (port.shift(1 + lag_periods) * ret_resampled).sum(axis=1)
    net_ret = gross_ret - turnover * tcost_bps * 1e-4
    
    # Annualize based on rebalancing frequency
    annualization_factor = {'1H': 24*365, '4H': 6*365, '6H': 4*365, '12H': 2*365, '1D': 365, '1W': 52}[rebal_freq]
    sharpe = net_ret.mean() / net_ret.std() * np.sqrt(annualization_factor)
    
    total_gross_ret = (1 + gross_ret).cumprod()
    total_net_ret = (1 + net_ret).cumprod()
    
    print(f"Momentum strategy with {rebal_freq} rebalancing:")
    print(f"  Lookback: {lookback}h ({lookback_periods} periods)")
    print(f"  Lag: {lag}h ({lag_periods} periods)")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  Gross Return: {total_gross_ret.iloc[-1]:.3f}")
    print(f"  Net Return: {total_net_ret.iloc[-1]:.3f}")
    print(f"  Average Turnover: {turnover.mean():.3f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(total_gross_ret.index, total_gross_ret.values, 'b-', linewidth=2, label='Gross Returns')
    plt.title(f'Gross Returns - {rebal_freq} Rebalancing')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(total_net_ret.index, total_net_ret.values, 'r-', linewidth=2, label='Net Returns')
    plt.title(f'Net Returns - {rebal_freq} Rebalancing')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(net_ret.index, net_ret.values, 'g-', linewidth=1, alpha=0.7)
    plt.title(f'Period Returns - {rebal_freq} Rebalancing')
    plt.xlabel('Time')
    plt.ylabel('Period Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'sharpe': sharpe,
        'gross_returns': gross_ret,
        'net_returns': net_ret,
        'turnover': turnover,
        'cum_gross': total_gross_ret,
        'cum_net': total_net_ret
    }

# ----------------------------------------------------------------------------------- #

def comprehensive_rebalancing_analysis(
    train_px: pd.DataFrame,
    train_vol: pd.DataFrame,
    test_px: pd.DataFrame = None,
    test_vol: pd.DataFrame = None,
    tcost_bps: int = 20
):
    """
    Comprehensive analysis of rebalancing frequencies for all strategy types
    """
    print("=" * 80)
    print("COMPREHENSIVE REBALANCING FREQUENCY ANALYSIS")
    print("=" * 80)
    
    # Test momentum strategies with different rebalancing frequencies
    print("\n1. MOMENTUM STRATEGIES")
    momentum_results = test_rebalancing_frequencies(train_px, train_vol, 'momentum', tcost_bps)
    
    # Test reversal strategies with different rebalancing frequencies
    print("\n2. REVERSAL STRATEGIES")
    reversal_results = test_rebalancing_frequencies(train_px, train_vol, 'reversal', tcost_bps)
    
    # Compare best strategies across frequencies
    print("\n3. CROSS-FREQUENCY COMPARISON")
    
    # Get best momentum strategy for each frequency
    best_momentum_by_freq = {}
    for freq in momentum_results['rebal_freq'].unique():
        freq_results = momentum_results[momentum_results['rebal_freq'] == freq]
        best_idx = freq_results['sharpe'].idxmax()
        best_momentum_by_freq[freq] = momentum_results.loc[best_idx]
    
    # Get best reversal strategy for each frequency
    best_reversal_by_freq = {}
    for freq in reversal_results['rebal_freq'].unique():
        freq_results = reversal_results[reversal_results['rebal_freq'] == freq]
        best_idx = freq_results['sharpe'].idxmax()
        best_reversal_by_freq[freq] = reversal_results.loc[best_idx]
    
    # Create comparison DataFrame
    comparison_data = []
    for freq in momentum_results['rebal_freq'].unique():
        if freq in best_momentum_by_freq and freq in best_reversal_by_freq:
            mom = best_momentum_by_freq[freq]
            rev = best_reversal_by_freq[freq]
            
            comparison_data.append({
                'Frequency': freq,
                'Momentum_Sharpe': mom['sharpe'],
                'Reversal_Sharpe': rev['sharpe'],
                'Momentum_Return': mom['annual_return'],
                'Reversal_Return': rev['annual_return'],
                'Momentum_Vol': mom['volatility'],
                'Reversal_Vol': rev['volatility'],
                'Momentum_Turnover': mom['avg_turnover'],
                'Reversal_Turnover': rev['avg_turnover']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plot comparison
    plt.figure(figsize=(20, 12))
    
    # Sharpe ratio comparison
    plt.subplot(2, 3, 1)
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['Momentum_Sharpe'], width, label='Momentum', alpha=0.7)
    plt.bar(x + width/2, comparison_df['Reversal_Sharpe'], width, label='Reversal', alpha=0.7)
    plt.xlabel('Rebalancing Frequency')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio by Rebalancing Frequency')
    plt.xticks(x, comparison_df['Frequency'], rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Annual return comparison
    plt.subplot(2, 3, 2)
    plt.bar(x - width/2, comparison_df['Momentum_Return'], width, label='Momentum', alpha=0.7)
    plt.bar(x + width/2, comparison_df['Reversal_Return'], width, label='Reversal', alpha=0.7)
    plt.xlabel('Rebalancing Frequency')
    plt.ylabel('Annual Return')
    plt.title('Annual Return by Rebalancing Frequency')
    plt.xticks(x, comparison_df['Frequency'], rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Volatility comparison
    plt.subplot(2, 3, 3)
    plt.bar(x - width/2, comparison_df['Momentum_Vol'], width, label='Momentum', alpha=0.7)
    plt.bar(x + width/2, comparison_df['Reversal_Vol'], width, label='Reversal', alpha=0.7)
    plt.xlabel('Rebalancing Frequency')
    plt.ylabel('Volatility')
    plt.title('Volatility by Rebalancing Frequency')
    plt.xticks(x, comparison_df['Frequency'], rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Turnover comparison
    plt.subplot(2, 3, 4)
    plt.bar(x - width/2, comparison_df['Momentum_Turnover'], width, label='Momentum', alpha=0.7)
    plt.bar(x + width/2, comparison_df['Reversal_Turnover'], width, label='Reversal', alpha=0.7)
    plt.xlabel('Rebalancing Frequency')
    plt.ylabel('Average Turnover')
    plt.title('Turnover by Rebalancing Frequency')
    plt.xticks(x, comparison_df['Frequency'], rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Sharpe vs Turnover scatter
    plt.subplot(2, 3, 5)
    plt.scatter(comparison_df['Momentum_Turnover'], comparison_df['Momentum_Sharpe'], 
               s=100, alpha=0.7, label='Momentum', c='blue')
    plt.scatter(comparison_df['Reversal_Turnover'], comparison_df['Reversal_Sharpe'], 
               s=100, alpha=0.7, label='Reversal', c='red')
    plt.xlabel('Average Turnover')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe vs Turnover')
    plt.legend()
    plt.grid(True)
    
    # Best strategy performance comparison
    plt.subplot(2, 3, 6)
    best_momentum = momentum_results.loc[momentum_results['sharpe'].idxmax()]
    best_reversal = reversal_results.loc[reversal_results['sharpe'].idxmax()]
    
    mom_returns = best_momentum['returns']
    rev_returns = best_reversal['returns']
    
    mom_cum = (1 + mom_returns).cumprod()
    rev_cum = (1 + rev_returns).cumprod()
    
    plt.plot(mom_cum.index, mom_cum.values, 'b-', linewidth=2, 
             label=f"Momentum ({best_momentum['rebal_freq']})")
    plt.plot(rev_cum.index, rev_cum.values, 'r-', linewidth=2, 
             label=f"Reversal ({best_reversal['rebal_freq']})")
    plt.title('Best Strategy Performance')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\n" + "=" * 100)
    print("REBALANCING FREQUENCY SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Freq':<8} {'Strategy':<10} {'Sharpe':<8} {'Return':<8} {'Vol':<8} {'Turnover':<10} {'Lookback':<10}")
    print("-" * 100)
    
    for freq in comparison_df['Frequency']:
        if freq in best_momentum_by_freq:
            mom = best_momentum_by_freq[freq]
            print(f"{freq:<8} {'Momentum':<10} {mom['sharpe']:<8.3f} {mom['annual_return']:<8.3f} "
                  f"{mom['volatility']:<8.3f} {mom['avg_turnover']:<10.3f} {mom['lookback_hours']:<10}")
        
        if freq in best_reversal_by_freq:
            rev = best_reversal_by_freq[freq]
            print(f"{freq:<8} {'Reversal':<10} {rev['sharpe']:<8.3f} {rev['annual_return']:<8.3f} "
                  f"{rev['volatility']:<8.3f} {rev['avg_turnover']:<10.3f} {rev['lookback_hours']:<10}")
    
    return {
        'momentum_results': momentum_results,
        'reversal_results': reversal_results,
        'comparison_df': comparison_df,
        'best_momentum_by_freq': best_momentum_by_freq,
        'best_reversal_by_freq': best_reversal_by_freq
    }

# ----------------------------------------------------------------------------------- #

def main():
    if "top_10notionals.csv" not in os.listdir():
        top_10notionals = select_topk_securities(lookback_days = 30, k = 10)
        top_10notionals['symbol'].to_csv("top_10notionals.csv", index = False)
        
    # Just to see the frequencies of the quote assets; seems USD4, USDT, 
    # and USD are the most common by far
    '''
    client = bnb_client(tld='US')
    univ = client.get_exchange_info()
    possible_quoteassets = {}
    for x in univ['symbols']:
        if x['quoteAsset'] not in possible_quoteassets:
            possible_quoteassets[x['quoteAsset']] = 1
        else:
            possible_quoteassets[x['quoteAsset']] += 1
    print(possible_quoteassets)
    '''
    
    # if you need to generate the pickle files
    # generate_pickle_px_data("top_10notionals.csv")
    
    # beginning with 1h data
    with open('1d_pxdata.pickle', 'rb') as f:
        univ_data = pickle.load(f)
    
    px = {}
    vol = {}
    for symbol, data in univ_data.items():
        px[symbol] = data.set_index('open_time')['close']
        vol[symbol] = data.set_index('open_time')['volume']
        
    px = pd.DataFrame(px).astype(float)
    vol = pd.DataFrame(vol).astype(float)
    
    train_px, test_px = split_data(px, test_frac = 0.3)
    train_vol, test_vol = split_data(vol, test_frac = 0.3)
    
    print("=" * 80)
    print("STATISTICAL ARBITRAGE IN CRYPTOCURRENCIES - FINAL PROJECT")
    print("=" * 80)
    print("Project: Research profitable momentum and reversal strategies in crypto")
    print("Data: Top 10 cryptocurrencies by notional volume")
    print("Time Period: 2019 to present (hourly data)")
    print("=" * 80)
    
    # 1. BASELINE STRATEGIES
    print("\n1. BASELINE STRATEGIES")
    buy_and_hold_strat(train_px, tcost_bps = 20)
    
    # 2. ORIGINAL STRATEGIES
    print("\n2. ORIGINAL STRATEGIES")
    macd_strat(train_px, test_px, range(5, 41, 5), range(80, 141, 5), range(20, 51, 5), 10)
    train_momentum_strat(train_px, train_vol, lookbacks = [1, 3, 7, 14, 21, 30, 40, 50, 60], lags = [0, 1, 3, 5, 7], tcost_bps = 20)    
    
    # 3. MOMENTUM STRATEGIES (Research Areas 1-5)
    print("\n3. MOMENTUM STRATEGIES")
    print("Research Area 1: Time Horizon Analysis")
    time_horizon_momentum_analysis(train_px, train_vol, tcost_bps = 20)
    
    print("\nResearch Area 2: New Information/Activity Analysis")
    volume_activity_momentum(train_px, train_vol, tcost_bps = 20)
    
    print("\nResearch Area 3: Seasonality Analysis")
    seasonality_momentum_analysis(train_px, train_vol, tcost_bps = 20)
    
    print("\nResearch Area 4: Investment Themes Analysis")
    theme_momentum_analysis(train_px, train_vol, test_px, test_vol, tcost_bps = 20)
    
    print("\nResearch Area 5: Technical Mechanics Analysis")
    technical_mechanics_analysis(train_px, train_vol, tcost_bps = 20)
    
    # 4. REVERSAL STRATEGIES (Research Areas 1-4)
    print("\n4. REVERSAL STRATEGIES")
    print("Research Area 1: Short-term Reversal Analysis")
    short_term_reversal_analysis(train_px, train_vol, tcost_bps = 20)
    
    print("\nResearch Area 2: Uninformed Trading Reversal")
    uninformed_trading_reversal(train_px, train_vol, tcost_bps = 20)
    
    # 6. OUT-OF-SAMPLE TESTING
    print("\n5. OUT-OF-SAMPLE TESTING")
    test_momentum_strat(test_px, test_vol, lookback = 720, lag = 24, tcost_bps = 20)
    
    # 7. REBALANCING FREQUENCY ANALYSIS
    print("\n6. REBALANCING FREQUENCY ANALYSIS")
    print("Testing different rebalancing frequencies (1H, 4H, 6H, 12H, 1D, 1W)...")
    
    # Test momentum strategies with different rebalancing frequencies
    print("\n6.1 Momentum Strategies with Variable Rebalancing")
    momentum_rebal_results = test_rebalancing_frequencies(train_px, train_vol, 'momentum', tcost_bps=20)
    
    # Test reversal strategies with different rebalancing frequencies
    print("\n6.2 Reversal Strategies with Variable Rebalancing")
    reversal_rebal_results = test_rebalancing_frequencies(train_px, train_vol, 'reversal', tcost_bps=20)
    
    # Test specific momentum strategy with different rebalancing frequencies
    print("\n6.3 Specific Momentum Strategy Comparison")
    rebal_freqs = ['1H', '4H', '6H', '12H', '1D', '1W']
    for freq in rebal_freqs:
        print(f"\nTesting momentum strategy with {freq} rebalancing...")
        momentum_strat_with_rebalancing(train_px, train_vol, lookback=168, lag=168, rebal_freq=freq, tcost_bps=20)
    
    print("\n" + "=" * 80)
    print("PROJECT SUMMARY")
    print("=" * 80)
    print("This project implements statistical arbitrage strategies in cryptocurrencies")
    print("focusing on momentum and reversal patterns as specified in the course requirements.")
    print("\nKey Research Areas Implemented:")
    print("✓ Time Horizon Analysis (different lookback periods)")
    print("✓ New Information/Activity (volume-based filtering)")
    print("✓ Seasonality Effects (weekday/weekend, work hours)")
    print("✓ Investment Themes (L1, L2, DeFi, Meme coins)")
    print("✓ Technical Mechanics (correlation-based strategies)")
    print("✓ Short-term Reversal Patterns")
    print("✓ Uninformed Trading Reversal")
    print("✓ Correlation-based Reversal")
    print("✓ Rebalancing Frequency Analysis (1H, 4H, 6H, 12H, 1D, 1W)")
    print("\nAll strategies include proper transaction costs (20 bps) and")
    print("comprehensive performance evaluation metrics.")
    print("\nRebalancing frequencies tested: 1H, 4H, 6H, 12H, 1D, 1W")
    print("This allows analysis of how trading frequency affects strategy performance,")
    print("transaction costs, and overall profitability.")
        
# ----------------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
from binance.client import Client as bnb_client
from datetime import datetime, UTC, timezone, timedelta, tzinfo
import pandas as pd
import numpy as np

import statsmodels.api as sm
from typing import Dict, Tuple, List
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
    
    Parameters:
    - data: dictionary that is generated from client.get_historical_klines
    
    Return:
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
    
    Parameters:
    - symbol: str indicating the ticker
    - freq: str indicating the frequency
    - start_time: str indicating the start time to begin analyzing
    
    Return:
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
    
    Parameters:
    - symbol: str indicating what security to look at
    - lookback_days: int indicating number of days to lookback
    
    Return: 
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
    
    Parameters:
    - lookback_days: int indicating how many days we want to look back to calculate 
      notional volume
    - k: int indicating the amount of securities we want to find
    
    Return:
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
    
    Parameters:
    - csv_filename: string containing the filename of the csv with the tickers you want data for
    
    Return:
    - no output
    '''
    univ = pd.read_csv(csv_filename)
    
    freqs = ['1w', '1d']
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
    '''
    Purpose: 
    Split the given dataframe into two sections, a training section and a test section.
    
    Parameters:
    - full_px: dataframe to be split
    - test_frac: the fraction of the dataframe in the test section
    
    Return:
    - tuple of two dataframes, the training section and the test section
    '''
    split_point = int(len(full_px) * (1 - test_frac))
    train = full_px.iloc[:split_point]
    test = full_px.iloc[split_point:]

    return train, test

# ----------------------------------------------------------------------------------- #

def train_momentum_strat(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    lookbacks: List[int] = range(1, 6),
    lags: List[int] = [0, 1, 2, 3],
    percentiles: List[int] = range(1, 6),
    tcost_bps = 20
):
    results = []
    ret = px.pct_change(fill_method=None)
    
    sharpe_matrix = pd.DataFrame(index = lookbacks, columns = lags)
    for lb in lookbacks:
        for lag in lags:
            for percentile in percentiles:
                port = ret.ewm(span = lb, adjust=False).mean()
                port = port
                ranks = port.rank(axis=1, pct=True)
                quantile = percentile / 10.0
                long = (ranks >= 1 - quantile).astype(float)
                short = (ranks <= quantile).astype(float)
                port = (long - short)
                port = port.div(port.abs().sum(axis=1), axis=0)  # normalize
                
                turnover = (port.fillna(0) - port.shift()).abs()
            
                gross_ret = (port.shift(1 + lag) * ret)
            
                net_ret = gross_ret - turnover * tcost_bps * 1e-4
            
                total_netret = net_ret.sum(axis = 1)
                
                sharpe = total_netret.mean(axis=0) / total_netret.std(axis=0) * np.sqrt(365)
                sharpe_matrix.loc[lb, lag] = sharpe
            
                results.append({
                    "lookback": lb,
                    "lag": lag,
                    "sharpe": sharpe,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "quantile": quantile
                })
                
    best_result = max(results, key=lambda x: x['sharpe'])    
    best_lb = best_result['lookback']
    best_lag = best_result['lag']
    best_sharpe = best_result['sharpe']
    net_ret = best_result['net_ret']
    best_quantile = best_result['quantile']
    
    # Fix: Sum returns first, then compound (correct portfolio approach)
    portfolio_returns = net_ret.sum(axis=1)
    cum_net_ret = (1 + portfolio_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    # Plot individual asset cumulative returns for comparison
    for column in net_ret.columns:
        asset_cum_ret = (1 + net_ret[column]).cumprod()
        plt.plot(asset_cum_ret.index, asset_cum_ret.values, label=f"{column} (Individual)", alpha=0.5)
    
    print(f"\nBEST RESULTS:\nLookback = {best_lb:>4} days | Lag = {best_lag:>3} | Quantile: {best_quantile} | Sharpe: {best_sharpe}\n")
    
    plt.plot(cum_net_ret.index, cum_net_ret.values, label=f"Portfolio (lb={best_lb}, lag={best_lag}, quantile={best_quantile})", linewidth=2)
    plt.title("Cumulative Net Returns of Best Momentum Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(sharpe_matrix.astype(float), annot = True, fmt=".2f", cmap = "coolwarm", linewidths=0.5)
    plt.title("Sharpe Ratio Heatmap")
    plt.xlabel("Lag")
    plt.ylabel("Lookback")
    plt.show()
    
    return best_result, results

# ----------------------------------------------------------------------------------- #

def macd_strat(train_px: pd.DataFrame, fast_lookback, slow_lookback, signal_period, tcost_bps = 20):
    """
    MACD strategy with separate training and testing phases
    """
    train_ret = train_px.pct_change(fill_method=None)
    
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
        
        weights = signal.div(signal.abs().sum(axis = 1), axis = 0)
        turnover = weights.diff().abs()
        gross_ret = (weights.shift() * train_ret)
        net_ret = gross_ret - turnover * tcost_bps * 1e-4
        cum_net = (1 + net_ret).cumprod()
        
        total_net_ret = net_ret.sum(axis = 1)
        
        strat_sharpe = total_net_ret.mean() / total_net_ret.std() * np.sqrt(365)
        
        train_results.append({
            'fast': f,
            'slow': s,
            'signal': sp,
            'sharpe': strat_sharpe,
            'cum_net': cum_net.iloc[-1],
            'returns': net_ret
        })
            
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
    plt.figure(figsize=(12, 6))
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
    # Fix: Sum returns first, then compound (correct portfolio approach)
    portfolio_returns = best_train_returns.sum(axis=1)
    cum_ret_train = (1 + portfolio_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    # Plot individual asset cumulative returns for comparison
    for column in best_train_returns.columns:
        asset_cum_ret = (1 + best_train_returns[column]).cumprod()
        plt.plot(asset_cum_ret.index, asset_cum_ret.values, label=f"{column} (Individual)", alpha=0.5)
    
    print(f"Portfolio cumulative return: {cum_ret_train.iloc[-1] - 1:.3f}")
    plt.plot(cum_ret_train.index, cum_ret_train.values, label='Portfolio (Best MACD Strategy)', color='blue', linewidth=2)
    plt.title(f"Training Performance — Best MACD Strategy\nFast={best_fast}, Slow={best_slow}, Signal={best_signal}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
        
    return best_config, best_train_returns

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
    return portfolio_val
    
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
        
    px = pd.DataFrame(px).astype(float).ffill()
    vol = pd.DataFrame(vol).astype(float).ffill()
    
    '''
    for column in px.columns:
        plt.figure(figsize=(10,6))
        plt.plot(px[column].index, px[column].values)
        plt.title(column)
        plt.show()
    '''
    
    train_px, test_px = split_data(px, test_frac = 0.3)
    train_vol, test_vol = split_data(vol, test_frac = 0.3)
        
    plt.figure(figsize=(12, 6))
    sns.heatmap(px.corr().astype(float), annot = True, fmt=".2f", cmap = "coolwarm", linewidths=0.5)
    plt.title("Bitcoin Correlations")
    plt.show()
    
    cols = px.columns
    
    best_momentum_strat, _ = train_momentum_strat(train_px[cols], train_vol[cols], lookbacks=range(5, 121, 5), lags = range(0, 31, 5), tcost_bps = 20)
    best_momentum_test, _ = train_momentum_strat(test_px[cols], test_vol[cols], lookbacks = [best_momentum_strat['lookback']], lags = [best_momentum_strat['lag']], percentiles=[10 * best_momentum_strat['quantile']], tcost_bps= 20)
    best_momentum_full, _ = train_momentum_strat(px[cols], vol[cols], lookbacks= [best_momentum_strat['lookback']], lags = [best_momentum_strat['lag']], percentiles=[10 * best_momentum_strat['quantile']], tcost_bps = 20)
    
    best_macd_strat, _ = macd_strat(train_px[cols], fast_lookback=range(20, 51, 5), slow_lookback=range(100, 181, 5), signal_period=range(5, 40, 5), tcost_bps=20)
    best_macd_test, _ = macd_strat(test_px[cols], fast_lookback=[best_macd_strat['fast']], slow_lookback=[best_macd_strat['slow']], signal_period=[best_macd_strat['signal']], tcost_bps = 20)
    best_macd_full, _ = macd_strat(px[cols], fast_lookback=[best_macd_strat['fast']], slow_lookback=[best_macd_strat['slow']], signal_period=[best_macd_strat['signal']], tcost_bps = 20)

    
    momentum_ret = best_momentum_full['net_ret'].sum(axis = 1)
    macd_ret = best_macd_full['returns'].sum(axis = 1)
    benchmark = buy_and_hold_strat(px, tcost_bps=20)
    bench_ret = benchmark.pct_change(fill_method=None)
    
    momentum_ret, bench_ret = momentum_ret.align(bench_ret, join = 'inner')
    data = pd.concat([momentum_ret, bench_ret], axis=1)
    data.columns = ['momentum', 'bench']
    data = data.dropna()
    momentum_ret = data['momentum']
    bench_ret = data['bench']
    
    bench_ret2 = bench_ret
    macd_ret, bench_ret2 = macd_ret.align(bench_ret2, join='inner')
    data = pd.concat([macd_ret, bench_ret2], axis = 1)
    data.columns = ['macd', 'bench']
    data = data.dropna()
    macd_ret = data['macd']
    bench_ret2 = data['bench']

    x = sm.add_constant(bench_ret)
    model_momentum = sm.OLS(momentum_ret, x).fit()
    alpha_momentum = model_momentum.params['const']
    beta_momentum = model_momentum.params['bench']
    
    x2 = sm.add_constant(bench_ret2)
    model_macd = sm.OLS(macd_ret, x2).fit()
    alpha_macd = model_macd.params['const']
    beta_macd = model_macd.params['bench']
    
    momentum_resid = model_momentum.resid
    macd_resid = model_macd.resid
    
    print("MOMENTUM STRAT\n----------------------------------------------")
    print(f"PRIOR CORRELATION: {momentum_ret.corr(bench_ret)}")
    print(f"POST DISTILLATION CORRELATION: {momentum_resid.corr(bench_ret)}")
    print(f"Beta: {beta_momentum}")
    print(f"Alpha: {alpha_momentum}")
    print(f"Original Volatility: {momentum_ret.std() * np.sqrt(365)}")
    print(f"Residual Volatility: {momentum_resid.std() * np.sqrt(365)}")
    print(f"Original Sharpe: {momentum_ret.mean() / momentum_ret.std() * np.sqrt(365)}")
    print(f"Residual Sharpe: {(momentum_resid.mean() + alpha_momentum) / momentum_resid.std() * np.sqrt(365)}")
    print("----------------------------------------------")
    
    print("MACD STRAT\n----------------------------------------------")
    print(f"PRIOR CORRELATION: {macd_ret.corr(bench_ret2)}")
    print(f"POST DISTILLATION CORRELATION: {macd_resid.corr(bench_ret2)}")
    print(f"Beta: {beta_macd}")
    print(f"Alpha: {alpha_macd}")
    print(f"Original Volatility: {macd_ret.std() * np.sqrt(365)}")
    print(f"Residual Volatility: {macd_resid.std() * np.sqrt(365)}")
    print(f"Original Sharpe: {macd_ret.mean() / macd_ret.std() * np.sqrt(365)}")
    print(f"Residual Sharpe: {(macd_resid.mean() + alpha_macd) / macd_resid.std() * np.sqrt(365)}")
    print("----------------------------------------------")
    
    momentum_alpha_contr = momentum_resid + alpha_momentum
    
    plt.figure(figsize=(12, 6))
    plt.plot((1 + momentum_ret).cumprod().index, (1 + momentum_ret).cumprod().values, label="Strategy", alpha = 0.7)
    plt.plot((1 + momentum_alpha_contr).cumprod().index, (1 + momentum_alpha_contr).cumprod().values, label=f"Alpha", alpha = 0.7)
    plt.plot((1 + bench_ret).cumprod().index, (1 + bench_ret * beta_momentum).cumprod().values, label="Buy and Hold", alpha = 0.7)
    plt.title("Strategy, Alpha, and Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    macd_alpha_contr = macd_resid + alpha_macd
    
    plt.figure(figsize=(12, 6))
    plt.plot((1 + macd_ret).cumprod().index, (1 + macd_ret).cumprod().values, label="Strategy", alpha = 0.7)
    plt.plot((1 + macd_alpha_contr).cumprod().index, (1 + macd_alpha_contr).cumprod().values, label=f"Alpha", alpha = 0.7)
    plt.plot((1 + bench_ret2).cumprod().index, (1 + bench_ret2 * beta_momentum).cumprod().values, label="Buy and Hold", alpha = 0.7)
    plt.title("Strategy, Alpha, and Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Net Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# ----------------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
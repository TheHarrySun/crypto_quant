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
            vol_avg = (vol.rolling(lb, min_periods=1).mean())
            port = (ret.rolling(lb, min_periods=1).mean()).rank(1) 
            port = port * np.log1p(vol_avg)
            
            port = port.sub(port.mean(axis=1), axis = 0)
            port = port.div(port.abs().sum(axis=1), axis = 0)
        
            turnover = (port.fillna(0) - port.shift()).abs().sum(axis = 1)
        
            gross_ret = (port.shift(1 + lag) * ret).sum(axis = 1)
        
            net_ret = gross_ret - turnover * tcost_bps * 1e-4
        
            sharpe = net_ret.mean(axis=0) / net_ret.std(axis=0) * np.sqrt(24 * 365)
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
            
            
            if (sharpe > -0.45):
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
    
    sharpe = net_ret.mean(axis = 0) / net_ret.std(axis = 0) * np.sqrt(365 * 24)
    
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
    with open('1h_pxdata.pickle', 'rb') as f:
        univ_data = pickle.load(f)
    
    px = {}
    vol = {}
    for symbol, data in univ_data.items():
        px[symbol] = data.set_index('open_time')['close']
        vol[symbol] = data.set_index('open_time')['volume']
        
    px = pd.DataFrame(px).astype(float)
    vol = pd.DataFrame(vol).astype(float)
    
    train_px, test_px = split_data(px, test_frac = 0.2)
    train_vol, test_vol = split_data(vol, test_frac = 0.2)
    
    train_momentum_strat(train_px, train_vol, lookbacks = [276, 504, 720, 1080], lags = [4, 8, 12, 24, 72],tcost_bps = 20)    
    
    test_momentum_strat(test_px, test_vol, lookback = 720, lag = 24, tcost_bps = 20)
        
# ----------------------------------------------------------------------------------- #

if __name__ == "__main__":
    main()
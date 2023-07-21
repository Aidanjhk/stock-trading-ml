from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import json
import argparse


def save_dataset(symbol, time_window):
    #credentials = json.load(open('creds.json', 'r'))
    api_key = "FLBKFNSA95C7STE0"
    print(symbol, time_window)
    ts = TimeSeries(key=api_key, output_format='pandas')
    if time_window == 'intraday':
        data, meta_data = ts.get_intraday(
            symbol=symbol, interval='5min', outputsize='full')
    elif time_window == 'daily':
        data, meta_data = ts.get_daily(symbol, outputsize='full')
    elif time_window == 'daily_adj':
        data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')

    pprint(data.head(10))

    data.to_csv(f'./{symbol}_{time_window}.csv')


if __name__ == "__main__":
    
    save_dataset("AAPL","intraday")

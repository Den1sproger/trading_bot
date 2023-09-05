import time
import json

import talib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



FILENAME_JSON_REVERSE = r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\creating_training_sample\reverse_fxcm.json'
FILENAME_JSON = r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\creating_training_sample\normal_fxcm.json'



def get_ema(close_prices: list[float]) -> list[float]:
    ema_50_l = []
    ema_150_l = []

    for candle in range(150, 300):
        prices = np.array(close_prices[:candle])
        ema_50 = talib.EMA(prices, timeperiod=50)
        ema_150 = talib.EMA(prices, timeperiod=150)
        ema_50 = float("%.7f" % ema_50[-1])
        ema_150 = float("%.7f" % ema_150[-1])
        ema_50_l.append(ema_50)
        ema_150_l.append(ema_150)

    return ema_50_l, ema_150_l


def get_reverse_price(max_point: float, price: float) -> float:
    # formula of the one reverse price
    return 2 * max_point - price


def create_reverse(data: dict[list[float]]) -> dict[list[float]]:
    # get a full reverse data
    
    reverse_data = {}

    full_close_prices = data['close149'] + data['close']
    max_point = max(full_close_prices)
    
    # candle's prices iteration
    for price_type, prices in list(data.items())[4:]:

        current_prices = []
        for price in prices:

            reverse_price = get_reverse_price(max_point, price)
            current_prices.append(reverse_price)

        reverse_data[price_type] = current_prices

    # left close prices iteration for compute ema
    left_close_prices_reversed = []
    for price in data['close149']:
        reverse_price = get_reverse_price(max_point, price)
        left_close_prices_reversed.append(reverse_price)

    ema_50, ema_150 = get_ema(left_close_prices_reversed + reverse_data['close'])

    reverse_data['ema_50'] = ema_50
    reverse_data['ema_150'] = ema_150
    
    with open(FILENAME_JSON_REVERSE, 'w', encoding='utf-8') as file:
        json.dump(reverse_data, file, ensure_ascii=False, indent=4)

    return reverse_data


def show_graph(data: dict[list[float, int]]):
    prices = {
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close']
    }
    stock_prices = pd.DataFrame(prices)
    
    plt.figure()
    
    up = stock_prices[stock_prices.close >= stock_prices.open]
    down = stock_prices[stock_prices.close < stock_prices.open]
    
    col1 = 'green'
    col2 = 'red'

    width = 0.8
    width2 = 0.15
    
    # Plotting up prices of the stock
    plt.bar(up.index, up.close-up.open, width, bottom=up.open, color=col1)
    plt.bar(up.index, up.high-up.close, width2, bottom=up.close, color=col1)
    plt.bar(up.index, up.low-up.open, width2, bottom=up.open, color=col1)
    
    # Plotting down prices of the stock
    plt.bar(down.index, down.close-down.open, width, bottom=down.open, color=col2)
    plt.bar(down.index, down.high-down.open, width2, bottom=down.open, color=col2)
    plt.bar(down.index, down.low-down.close, width2, bottom=down.close, color=col2)
    
    # rotating the x-axis tick labels at 30degree 
    # towards right
    plt.xticks(rotation=30, ha='right')

    ema = {
        'ema_50': data['ema_50'],
        'ema_150': data['ema_150']
    }
    stock_ema = pd.DataFrame(ema)

    col_ema_50 = 'blue'
    col_ema_150 = 'olive'
    
    plt.plot(stock_ema.ema_50, color=col_ema_50)
    plt.plot(stock_ema.ema_150, color=col_ema_150)

    plt.show()
# def show_graph(data: dict[list[float, int]]) -> None:
#     # DataFrame to represent opening , closing, high 
#     # and low prices of a stock for a week
#     stock_prices = pd.DataFrame(data)
    
#     plt.figure()
    
#     # "up" dataframe will store the stock_prices 
#     # when the closing stock price is greater
#     # than or equal to the opening stock prices
#     up = stock_prices[stock_prices.close >= stock_prices.open]
    
#     # "down" dataframe will store the stock_prices
#     # when the closing stock price is
#     # lesser than the opening stock prices
#     down = stock_prices[stock_prices.close < stock_prices.open]
    
#     # When the stock prices have decreased, then it
#     # will be represented by blue color candlestick
#     col1 = 'green'
    
#     # When the stock prices have increased, then it 
#     # will be represented by green color candlestick
#     col2 = 'red'
    
#     # Setting width of candlestick elements
#     width = 0.8
#     width2 = 0.15
    
#     # Plotting up prices of the stock
#     plt.bar(up.index, up.close-up.open, width, bottom=up.open, color=col1)
#     plt.bar(up.index, up.high-up.close, width2, bottom=up.close, color=col1)
#     plt.bar(up.index, up.low-up.open, width2, bottom=up.open, color=col1)
    
#     # Plotting down prices of the stock
#     plt.bar(down.index, down.close-down.open, width, bottom=down.open, color=col2)
#     plt.bar(down.index, down.high-down.open, width2, bottom=down.open, color=col2)
#     plt.bar(down.index, down.low-down.close, width2, bottom=down.close, color=col2)
    
#     # rotating the x-axis tick labels at 30degree 
#     # towards right
#     plt.xticks(rotation=30, ha='right')
    
#     # displaying candlestick chart of stock data 
#     # of a week
#     plt.show()



def get_data_from_json() -> dict[list[float]]:
    filepath = r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\creating_training_sample\b2.json'
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    full_close_prices = data['close149'] + data['close']
    ema_50, ema_150 = get_ema(full_close_prices)
    data_with_ema = {
        'open': data['open'],
        'high': data['high'],
        'low': data['low'],
        'close': data['close'],
        'ema_50': ema_50,
        'ema_150': ema_150
    }

    with open(FILENAME_JSON, 'w', encoding='utf-8') as file:
        json.dump(data_with_ema, file, ensure_ascii=False, indent=4)

    return data, data_with_ema
    


if __name__ == '__main__':
    print('Привет, я нарисую тебе график\n')
    data, update_data = get_data_from_json()
    reverse_data = create_reverse(data)
    print('Draw normal graph')
    show_graph(update_data)
    print('Draw reverse graph')
    show_graph(reverse_data)
    input('Нажмите Enter чтобы выйти')
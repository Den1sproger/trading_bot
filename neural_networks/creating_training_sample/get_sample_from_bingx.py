import time
import json

import talib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



CANDLES = 299
FILENAME_JSON = '_bingx.json'
FILENAME_JSON_REVERSE = '_bingx_reverse.json'
# STANDART_CONTRACTS_URL = 'https://open-api.bingx.com/openApi/swap/v3/quote/klines'
FUTURES_URL = 'https://open-api.bingx.com/openApi/swap/v3/quote/klines?'



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


def get_reverse(max_point: float,
                close_prices_left: list[float],
                data_prices: dict[list[float]]) -> dict[list[float]]:
    # get a full reverse data

    reverse_data = {}

    # candle's prices iteration
    for price_type, prices in data_prices.items():

        current_prices = []
        for price in prices:

            reverse_price = get_reverse_price(max_point, price)
            current_prices.append(reverse_price)

        reverse_data[price_type] = current_prices

    # left close prices iteration for compute ema
    left_close_prices_reversed = []
    for price in close_prices_left:
        reverse_price = get_reverse_price(max_point, price)
        left_close_prices_reversed.append(reverse_price)

    ema_50, ema_150 = get_ema(left_close_prices_reversed + reverse_data['close'])

    reverse_data['ema_50'] = ema_50
    reverse_data['ema_150'] = ema_150
    
    return reverse_data


def parsing_to_json(response: dict[list[dict[int, float, str]]]) -> None:
    response = response['data']
    data = {}
    price_types = ['open', 'close', 'high', 'low']    # open price, high price, low price, close price

    for key in price_types:
        data[key] = [float(kline[key]) for kline in response[149:]]

    right_close = data['close']
    left_close = [float(kline['close']) for kline in response[:149]]       # close prices

    right_high = data['high']
    left_high = [float(kline['high']) for kline in response[:149]]         # high prices

    ema_50, ema_150 = get_ema(close_prices=left_close + right_close)

    data_reverse = get_reverse(
        max_point=max(left_high + right_high),
        close_prices_left=left_close,
        data_prices=data
    )

    data['ema_50'] = ema_50
    data['ema_150'] = ema_150

    with open(FILENAME_JSON_REVERSE, 'w', encoding='utf-8') as file:
        json.dump(data_reverse, file, ensure_ascii=False, indent=4)

    with open(FILENAME_JSON, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def draw_graph(filename: str):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

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
        
            
def main():
    # base_endpoint = get_api_url()
    symbol = input('Ввведите монету (Пример: BTC-USDT) -> ')
    interval = input('Введите таймфрейм (Пример: 1 минута - 1m, 15 минут - 15m, 1 час - 1h) -> ')
    endtime = input('Введите время начала последней свечи в формате Y-m-d H:M:S(Пример: 2022-05-14 12:34:00) -> ')

    try:
        unix_endtime = int(time.mktime(time.strptime(endtime, '%Y-%m-%d %H:%M:%S')))
    except ValueError:
        print('Время не соответствует формату')
        input('Press Enter to exit')
        return
    
    filename_data = endtime.replace(' ', '_').replace(':', '-')
    
    global FILENAME_JSON, FILENAME_JSON_REVERSE

    FILENAME_JSON = f"{interval}{symbol}{filename_data}{FILENAME_JSON}"
    FILENAME_JSON_REVERSE = f"{interval}{symbol}{filename_data}{FILENAME_JSON_REVERSE}"

    try:
        url = FUTURES_URL + f'symbol={symbol.upper()}&interval={interval}&endTime={unix_endtime}999&limit={CANDLES}'
        print(url)
        req = requests.get(url)
    except ValueError:
        print('Неверная монета и/или таймфрейм')
        input('Press Enter to exit')
        return

    # print(req.json())
    parsing_to_json(response=req.json())

    print('Рисую базовый график...')
    draw_graph(FILENAME_JSON)

    print('Рисую реверсивный график...')
    draw_graph(FILENAME_JSON_REVERSE)

    input('Press Enter to exit')


if __name__ == '__main__':
    main()
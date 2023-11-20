import time
import json
import logging

import websocket
import talib
import requests
import numpy as np

from binance_config import API_KEY, SECRET_KEY
from binance.client import Client



logging.basicConfig(
    level=logging.INFO,
    # filename='py_log.log',
    # filemode='w',
    format="%(asctime)s %(levelname)s %(message)s"
)


SYMBOL = 'BTCUSDT'
INTERVAL = '1m'


class Stream:
    """Streaming data about the latest prices via websockets"""
    

    def __init__(self, symbol: str, interval: str) -> None:
        self.symbol = symbol
        self.interval = interval

        self.closing_data = self.get_prices()     # list with the last prices
        self.client = Client(api_key=API_KEY, api_secret=SECRET_KEY)

        self.buy = False
        self.sell = True

        self.last_ema_50 = None
        self.last_ema_150 = None

        self.last_kline_start_time: str


    def create_request(func):
        def wrapper(self):
            url = f'https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval={self.interval}'
            r = requests.get(url)

            result = func(self, response=r)
            return result
        return wrapper
    

    @create_request
    def get_prices(self, **kwargs) -> list[list[float]]:
        # get the 3 lists with the last several hundreds
        # high prices, low prices, close prices 
        response = kwargs['response']
        self.last_kline_start_time = str(response.json()[-1][0])
        # logging.info(f'start: {self.last_kline_start_time=}')
        data = []
        for i in range(2, 5):    # high price, low price, close price
            item = [float(kline[i]) for kline in response.json()]
            data.append(item)

        return data   # prices
    

    @create_request
    def get_last_prices_for_update(self, **kwargs) -> list[float]:
        # get prices of the last closed candle
        response = kwargs['response']

        data = []
        for i in range(2, 5):    # high price, low price, close price
            item = float(response.json()[-2][i])
            data.append(item)

        return data


    def update_price_list(self, kline: dict, delete_index: int = -1) -> None:
        # replace the old last prices with the new last prices
        # and delete first or last price in lists
        
        if delete_index == -1:
            count = 0
            for i in ['h', 'l', 'c']:            # high price, low price, close price
                self.prices_data[count].pop(delete_index)
                self.prices_data[count].append(float(kline.get(i)))

                count += 1
                
        elif delete_index == 0:
            count = 0
            for i in ['h', 'l', 'c']:            # high price, low price, close price    
                prices = self.get_last_prices_for_update()
                self.prices_data[count].pop(-1)
                self.prices_data[count].append(prices[count])
                self.prices_data[count].pop(delete_index)
                self.prices_data[count].append(float(kline.get(i)))

                count += 1

        # self.get_info()

    def on_error(self, ws, error):
        print(error)

    def on_close(self, ws):             # for websockets
        print("### closed ###")

    def on_open(self, ws):
        print("### connected ###")

        
    def place_order(self, order_type: str) -> None:     # buy or sell
        if order_type.lower() == 'buy':
            order = self.client.create_order(symbol=SYMBOL, side='buy', type='MARKET')
            print('Open position', order)
        elif order_type.lower() == 'sell':
            order = self.client.create_order(symbol=SYMBOL, side='sell', type='MARKET')
            print('Close position', order)


    def on_message(self, ws, message):  # getting data from websockets
        return_data = json.loads(message)
        kline_start_time = str(return_data.get("k").get("t"))
        number_of_trades = str(return_data.get("k").get("n"))
        logging.info(f'\n{kline_start_time=}\n{number_of_trades=}')
        # self.closing_data.append(float(last_price))
        # self.closing_data.pop(0)

        # ema_50 = talib.EMA(np.array(self.closing_data), timeperiod=50)[-1]
        # ema_150 = talib.EMA(np.array(self.closing_data), timeperiod=150)[-1]

        # if (ema_50 < ema_150 and self.last_ema_50) and \
        #     (self.last_ema_50 > self.last_ema_150 and not self.buy):
        #     print('Buy!')
        #     self.place_order('buy')
        #     self.buy = True
        #     self.sell = False

        # elif (ema_50 > ema_150 and self.last_ema_50) and \
        #     (self.last_ema_50 > self.last_ema_150 and not self.sell):
        #     print('Sell!')
        #     self.place_order('sell')
        #     self.buy = False
        #     self.sell = True

        # self.last_ema_50 = ema_50
        # self.last_ema_150 = ema_150

        time.sleep(5)


    def get_data(self):
        url = f'wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}'
        wsa = websocket.WebSocketApp(
            url, on_message=self.on_message,
            on_error = self.on_error,
            on_close = self.on_close,
        )
        wsa.on_open = self.on_open
        wsa.run_forever()



if __name__ == '__main__':
    stream = Stream(SYMBOL, INTERVAL)
    stream.get_data()





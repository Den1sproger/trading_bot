import time
import json
import logging

import unicorn_binance_websocket_api
import talib
import requests
import numpy as np

from binance_config import API_KEY, SECRET_KEY
from binance.client import Client



# logging.basicConfig(
#     level=logging.INFO,
#     filename='py_log.log',
#     filemode='w',
#     format="%(asctime)s %(levelname)s %(message)s"
# )

SYMBOL = 'BTCUSDT'
INTERVAL = '1m'


class Trade:
    """Streaming data about the last, low and high prices via websockets"""
    

    def __init__(self, symbol: str, interval: str) -> None:
        self.symbol = symbol
        self.interval = interval

        self.prices_data = self.get_prices()     # lists with the last prices
        
        self.client = Client(api_key=API_KEY, api_secret=SECRET_KEY)

        self.price_move_high = []
        self.price_move_low = []

        self.buy = False
        self.sell = True

        self.last_ema_50 = None
        self.last_ema_150 = None

        self.last_slowk = None
        self.last_slowd = None

        self.last_number_of_trades = 0
        self.last_kline_start_time: str
        

    def create_request(func):
        def wrapper(self):
            url = f'https://fapi.binance.com/fapi/v1/klines?symbol={self.symbol}&interval={self.interval}'
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
        

    def place_order(self, order_type: str) -> None:     # buy or sell
        if order_type.lower() == 'buy':
            order = self.client.create_order(symbol=SYMBOL, side='buy', type='MARKET')
            print('Open position', order)
        elif order_type.lower() == 'sell':
            order = self.client.create_order(symbol=SYMBOL, side='sell', type='MARKET')
            print('Close position', order)


    def update_price_list(self, kline: dict, delete_index: int = -1) -> None:
        # replace the old last prices with the new last prices
        # and delete first or last price in lists
        def update(count: int, del_index: int, add_price: float) -> None:
            self.prices_data[count].pop(del_index)
            self.prices_data[count].append(add_price)
        
        if delete_index == -1:
            count = 0
            for i in ['h', 'l', 'c']:            # high price, low price, close price
                update(count, delete_index, add_price=float(kline.get(i)))
                count += 1
                
        elif delete_index == 0:
            count = 0
            for i in ['h', 'l', 'c']:            # high price, low price, close price    
                prices = self.get_last_prices_for_update()
                # self.prices_data[count].pop(-1)
                # self.prices_data[count].append(prices[count])
                update(count, del_index=-1, add_price=prices[count])
                update(count, delete_index, add_price=float(kline.get(i)))
                count += 1

        # self.get_info()


    def on_message(self, kline: dict) -> None:    # get data from websockets        
        # if a new candle has started
        kline_start_time = str(kline.get('t'))

        if kline_start_time == self.last_kline_start_time:
            # delete first price in lists and add new price in the list end
            self.update_price_list(kline)

        else:
            # delete last price in lists and add new price in the list end
            logging.info('new canbdle')
            self.update_price_list(kline, delete_index=0)
            self.last_kline_start_time = kline_start_time

        # ema 50 and ema 150
        ema_50 = talib.EMA(np.array(self.prices_data[2]), timeperiod=50)
        ema_150 = talib.EMA(np.array(self.prices_data[2]), timeperiod=150)
        ema_50 = "%.7f" % ema_50[-1]
        ema_150 = "%.7f" % ema_150[-1]
        print(f'50: {ema_50}')
        print(f'150: {ema_150}')

        # stochastic
        slowk, slowd = talib.STOCH(
            high=np.array(self.prices_data[0]),
            low=np.array(self.prices_data[1]),
            close=np.array(self.prices_data[2]),
            fastk_period=5, slowk_period=5, slowd_period=5
        )
        blue = "%.2f" % slowk[-1]
        orange = "%.2f" % slowd[-1]
        print(f'Blue: {blue}')
        print(f'Orange: {orange}\n')


    def streaming(self):
        ubwa = unicorn_binance_websocket_api.BinanceWebSocketApiManager(exchange="binance.com-futures")
        ubwa.create_stream([f'kline_{self.interval}'], [self.symbol])

        count = 0
        kline = {}
        while True:
            message = ubwa.pop_stream_data_from_stream_buffer()

            if message:
                if count == 0:
                    try:
                        data = json.loads(message)
                        kline = data['data']['k']
                    except KeyError: pass
                    else:
                        self.on_message(kline)
                        # time.sleep(5)
                elif count == 5: count = -1               
                count += 1
                 

    def __del__(self) -> None:
        return



if __name__ == '__main__':
    stream = Trade(SYMBOL, INTERVAL)
    stream.streaming()
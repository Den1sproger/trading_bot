import time
import json
import logging
import urllib3

import requests
import numpy as np
import pandas as pd
import keras

from datetime import datetime
from technical_analysis import overlays, indicators
from config import *


SYMBOL = 'BTC/USD'
INTERVAL = '1m'


class Trade:
    """Streaming data about the last, low and high prices via websockets"""
    

    def __init__(self, symbol: str, interval: str) -> None:
        self.symbol = symbol.strip()
        self.symbol_no_slash = self.symbol.replace('/', r'%2F')
        self.interval = interval.strip()
        self.market_url = 'https://dzengi.com/trading/platform/charting'

        self.prices_data = self.get_prices()     # lists with the last prices

        self.buy = False
        self.sell = True

        self.is_intersection = False
        self.candles_after_intersection = 0
        self.current_position = ''

        self.last_kline_start_time: str


    def __get_start_ema(self, close_prices: list[float]) -> list[float]:
        ema_50_l = []
        ema_150_l = []

        for candle in range(150, 300):
            prices = pd.Series(close_prices[:candle])
            ema_50 = overlays.ema(prices, period=50)
            ema_150 = overlays.ema(prices, period=150)
            ema_50_l.append(float(tuple(ema_50)[-1]))
            ema_150_l.append(float(tuple(ema_150)[-1]))

        return ema_50_l, ema_150_l


    def __calc_current_ema(self) -> list[float]:
        ema_50 = overlays.ema(price=pd.Series(self.prices_data[2]), period=50)
        ema_150 = overlays.ema(price=pd.Series(self.prices_data[2]), period=150)
        
        ema_50 = list(ema_50)[-2:]
        ema_150 = list(ema_150)[-2:]
        
        return ema_50, ema_150


    def __send_message(self, msg_text: str, retry: int = 5) -> None:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        url = f'https://api.telegram.org/bot{API_KEY}/sendMessage'

        ikb = json.dumps({
            'inline_keyboard': [[{
                'text': 'Рынок',
                'callback_data': '0',
                'url': self.market_url
            }]]
        })
        try:
            requests.post(
                url=url,
                timeout=5,
                verify=False,
                data={
                    'chat_id': int(GROUP_CHAT_ID),
                    'text': msg_text,
                    'reply_markup': ikb
                }
            )
        except Exception as _ex:
            if retry:
                logging.info(f"retry={retry} send_msg => {_ex}")
                retry -= 1
                time.sleep(5)
                self.__send_message(msg_text, retry)
            else:
                logging.info(f'Cannot send message to chat_id = {GROUP_CHAT_ID}')


    def __normalization(self) -> list[any]:
        data = []

        all_prices = self.prices_data[2] + self.prices_data[3] + self.prices_data[4]
        max_price = max(all_prices)
        min_price = min(all_prices)

        for price_index in (2, 3, 4):
            new_prices = []

            for price in self.prices_data[price_index]:
                new_price = (price - min_price) / (max_price - min_price)
                new_prices.append(float("%.10f" % new_price))

            data.append(new_prices)

        return data
    

    def __check_neural_network(self) -> str:
        model = keras.models.load_model(f'saved_models/{self.current_position}.keras')
        normalizated_data = self.__normalization()
        result = model.predict(np.array([normalizated_data]))

        if result[0][0] > result[0][1]:
            action = 'BUY'
        else:
            action = 'NOT_BUY'
        logging.info(f'nn decision => {action}')

        return action


    def __check_intersection_ema(self)-> bool:
        last_close_prices = self.prices_data[2][-10:]
        last_150_ema = self.prices_data[4][-10:]

        if self.current_position == 'short':
            max_last_ema = max(last_150_ema)
            max_last_close_price = max(last_close_prices)
            if max_last_close_price > max_last_ema:
                return True
            
        elif self.current_position == 'long':
            min_last_ema = min(last_150_ema)
            min_last_close_price = min(last_close_prices)
            if min_last_close_price < min_last_ema:
                return True
            
        return False


    def __check_intersection_stoch(self,
                                   blue: tuple[float],
                                   orange: tuple[float]) -> None:
        if (blue[-1] >= 77.5) and (orange[-1] >= 77.5) and \
            (blue[-1] < orange[-1]) and (blue[-2] > orange[-2]):
            self.current_position = 'short'
        
        elif (blue[-1] <= 22.5) and (orange[-1] <= 22.5) and \
            (blue[-1] > orange[-1]) and (blue[-2] < orange[-2]):
            self.current_position = 'long'


    def __check_stoch(self, blue: float, orange: float) -> str | None:
        if (blue >= 40) and (orange >= 40):
            return 'short'
        
        elif (blue <= 60) and (orange <= 60):
            return 'long'
    

    def create_request(candles):
        def decorator(func):
            def wrapper(self):
                url = f"https://api-adapter.dzengi.com/api/v1/klines?symbol={self.symbol_no_slash}&interval={self.interval}&limit={candles}"
                r = requests.get(url)

                result = func(self, response=r)
                return result
            return wrapper
        return decorator
    

    @create_request(candles=300)
    def get_prices(self, **kwargs) -> list[list[float]]:
        # get the 3 lists with the last several hundreds
        # high prices, low prices, close prices 
        response = kwargs['response']
        self.last_kline_start_time = str(response.json()[-1][0])
        data = []
        for i in range(2, 5):    # high price, low price, close price
            item = [float(kline[i]) for kline in response.json()[150:]]
            data.append(item)

        left_close = [float(kline[4]) for kline in response.json()[:150]]
        ema_50, ema_150 = self.__get_start_ema(left_close + data[-1])

        data.append(ema_50)
        data.append(ema_150)

        return data   # prices
    

    @create_request(candles=2)
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


    def on_message(self, kline: dict) -> None:    # get data from websockets        
        # if a new candle has started
        kline_start_time = str(kline.get('t'))

        if kline_start_time == self.last_kline_start_time:
            # delete first price in lists and add new price in the list end
            count = 0
            for i in ['h', 'l', 'c']:            # high price, low price, close price
                self.prices_data[count].pop(-1)
                self.prices_data[count].append(float(kline.get(i)))
                count += 1

            ema_50, ema_150 = self.__calc_current_ema()
            for item in (ema_50, ema_150):
                self.prices_data[count].pop(-1)
                self.prices_data[count].append(item[-1])
                count += 1
        else:
            # delete last price in lists and add new price in the list end
            count = 0
            for i in ['h', 'l', 'c']:            # high price, low price, close price    
                prices = self.get_last_prices_for_update()
                self.prices_data[count].pop(-1)
                self.prices_data[count].append(prices[count])
                self.prices_data[count].pop(0)
                self.prices_data[count].append(float(kline.get(i)))
                count += 1

            ema_50, ema_150 = self.__calc_current_ema()
            for item in (ema_50, ema_150):
                self.prices_data[count].pop(0)
                self.prices_data[count].pop(-1)
                self.prices_data[count] += item
                count += 1

            self.last_kline_start_time = kline_start_time

            if self.is_intersection:
                self.candles_after_intersection += 1
                if self.candles_after_intersection > 7:
                    self.candles_after_intersection = 0
                    self.current_position = ''
                    self.is_intersection = False


        slowk, slowd = indicators.stochastic(
            high=pd.Series(self.prices_data[0]),
            low=pd.Series(self.prices_data[1]),
            close=pd.Series(self.prices_data[2]),
            period=5, perc_k_smoothing=5, perc_d_smoothing=5
        )

        blue = tuple(slowk)[-2:]
        orange = tuple(slowd)[-2:]

        # print(f"[{round(blue[-2], 2)}] {round(blue[-1], 2)}")
        # print(f"[{round(orange[-2], 2)}] {round(orange[-1], 2)}\n")
        # print(f"[{self.prices_data[3][-2]}] {self.prices_data[3][-1]}\n")


        if not self.is_intersection:
            self.__check_intersection_stoch(blue, orange)
            if not self.current_position:
                return
            self.is_intersection = True
            logging.info(f'\nIntersecion stoch => {self.current_position}')
        else:
            stoch_position = self.__check_stoch(blue[-1], orange[-1])
            if stoch_position != self.current_position:
                logging.info(f'stoch position => {stoch_position}')
                self.is_intersection = False
                self.candles_after_intersection = 0
                self.current_position = ''
                return

        if self.__check_intersection_ema():
            logging.info(f'Intersecion ema => {self.current_position}')
            nn_action = self.__check_neural_network()
            if nn_action == 'BUY':
                logging.info(f'nn decision => {nn_action} ')
                unix_time = int(self.last_kline_start_time[:-3])
                candle_start_time = datetime.fromtimestamp(unix_time)

                self.__send_message(
                    f'❗️❗️❗️СИГНАЛ❗️❗️❗️\n{self.symbol}\n{self.interval}\n' \
                    f'{self.current_position}\nВремя начала свечи: {candle_start_time}'
                )



    def streaming(self):
        url = f'https://api-adapter.dzengi.com/api/v1/klines?symbol={self.symbol_no_slash}&interval={self.interval}&limit=1'

        while True:
            r = requests.get(url)
            response = r.json()[-1]

            kline = {
                't': response[0],
                'h': response[2],
                'l': response[3],
                'c': response[4]
            }
            self.on_message(kline)
            time.sleep(35)
                 

    def __del__(self) -> None:
        return



if __name__ == '__main__':
    symbol = input('Ввеите монету (Пример: BTC/USD) => ')
    interval = input('Введите таймфрейм (Примеры: 1m, 5m, 15m) => ')

    logging.basicConfig(
        level=logging.INFO,
        filename=f"log_{symbol.replace('/', '_')}_{interval}.log",
        filemode='w',
        format="%(asctime)s %(levelname)s %(message)s"
    )
    # symbol = SYMBOL
    # interval = INTERVAL

    stream = Trade(symbol=symbol.upper(), interval=interval.lower())
    stream.streaming()
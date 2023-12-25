import time
import json
import logging
import urllib3
import os

import requests
import numpy as np
import pandas as pd
import keras

from pathlib import Path
from datetime import datetime
from technical_analysis import overlays, indicators
from config import *
from draw_candle_graph import DrawGraph



SYMBOL = 'BTC/USD'
INTERVAL = '1m'
ROOT_DIR = Path(__file__).parent
CANDLE_IMAGE_DIR = 'saved_candle_images'

    


class Trade:
    """Streaming data about the last, low and high prices via websockets"""
    

    def __init__(self, symbol: str, interval: str) -> None:
        self.symbol = symbol.strip()
        self.symbol_no_slash = self.symbol.replace('/', r'%2F')
        self.interval = interval.strip()
        self.market_url = 'https://dzengi.com/trading/platform/charting'

        self.prices_data = self.get_prices()     # lists with the last prices
        self.buy = False

        self.is_intersection = False
        self.candles_after_intersection = 0
        self.current_position = ''

        self.last_kline_start_time: str
        # self.candles_count = 0



    def __reset_signal(self):
        self.candles_after_intersection = 0
        self.current_position = ''
        self.is_intersection = False



    def __stoch_is_valid(self,
                      blue: tuple[float],
                      orange: tuple[float]) -> bool:
        if not self.is_intersection:
            # checking on stoch interseciton for short position
            if (orange[-2] >= 75) \
                and (blue[-1] < orange[-1]) \
                and (blue[-2] > orange[-2]):
                self.current_position = 'short'

            # checking on stoch interseciton for long position
            elif (orange[-2] <= 25) \
                and (blue[-1] > orange[-1]) \
                and (blue[-2] < orange[-2]):
                self.current_position = 'long'

            if not self.current_position: return False
            logging.info(f'\n\nIntersecion stoch => {self.current_position}')
            self.is_intersection = True
            return True
        
        # checking acceptable values of stochastic
        stoch_position = None
        if (self.current_position == 'short') and (orange[-1] >= 25):
            stoch_position = 'short'
        elif (self.current_position == 'long') and (orange[-1] <= 75):
            stoch_position = 'long'
        
        # if the stochastic has gone beyond the acceptable values
        if stoch_position != self.current_position:
            logging.info(f'stoch position => {stoch_position}')
            self.__reset_signal()
            return self.__stoch_is_valid(blue, orange)    # check stoch again on this step
        
        # if there was an new intersecton in interval [25, 75]
        if (self.current_position == 'short' and blue[-2] > orange[-2] \
            and orange[-2] < 75 and blue[-2] < 75) or \
            (self.current_position == 'long' and blue[-2] < orange[-2] \
            and orange[-2] > 25 and blue[-2] > 25):
            logging.info(f'New stochastic intersection')
            self.__reset_signal()
            return False

        return True



    def __entry_point_ema_is_valid(self) -> bool:
        if self.current_position == 'short':
            ema_position_is_valid = self.prices_data[3][-1] < self.prices_data[4][-1]
            is_intersection_ema_150_bar = max(self.prices_data[0][-10:]) > max(self.prices_data[4][-10:])

            # last bar should be below the ema 50
            # and penultimate bar should be above the ema 50
            entry_point_is_valid = self.prices_data[2][-1] < self.prices_data[3][-1] and \
                (self.prices_data[2][-2] > self.prices_data[3][-2] \
                or self.prices_data[2][-3] > self.prices_data[3][-3])
            
        elif self.current_position == 'long':
            ema_position_is_valid = self.prices_data[3][-1] > self.prices_data[4][-1]
            is_intersection_ema_150_bar = min(self.prices_data[1][-10:]) < min(self.prices_data[4][-10:])

            # last bar should be above the ema 50
            # penultimate bar should be below the ema 50
            entry_point_is_valid = self.prices_data[2][-1] > self.prices_data[3][-1] and \
                (self.prices_data[2][-2] < self.prices_data[3][-2] \
                or self.prices_data[2][-3] < self.prices_data[3][-3])

        if ema_position_is_valid \
            and is_intersection_ema_150_bar \
            and entry_point_is_valid:
            return True

        return False



    def __check_trend(self) -> str:
        left_close_prices = self.prices_data[2][150:225]
        right_close_prices = self.prices_data[2][225:]
        maximums = (max(left_close_prices), max(right_close_prices))
        minimums = (min(left_close_prices), min(right_close_prices))

        if self.current_position == 'short':
            is_maximums_valid = maximums[0] > maximums[1]
            is_minimums_valid = minimums[0] > minimums[1]
        elif self.current_position == 'long':
            is_maximums_valid = maximums[0] < maximums[1]
            is_minimums_valid = minimums[0] < minimums[1]

        if is_maximums_valid and is_minimums_valid:
            logging.info('trend => BUY')
            return 'BUY'
        
        logging.info('trend => NOT_BUY')
        return 'NOT_BUY'



    def __get_start_ema(self, close_prices: list[float]) -> list[float]:
        prices = pd.Series(close_prices)
        ema_50 = overlays.ema(prices, period=50)
        ema_150 = overlays.ema(prices, period=150)
        ema_50_l = [round(i, 6) for i in list(ema_50)[150:]]
        ema_150_l = [round(i, 6) for i in list(ema_150)[150:]]

        return ema_50_l, ema_150_l



    def __calc_ema(self):
        ema_50 = overlays.ema(price=pd.Series(self.prices_data[2]), period=50)
        ema_150 = overlays.ema(price=pd.Series(self.prices_data[2]), period=150)

        self.prices_data[3] = [round(i, 6) for i in list(ema_50)[150:]]
        self.prices_data[4] = [round(i, 6) for i in list(ema_150)[150:]]



    def __send_message(self,
                       msg_text: str,
                       photo_path: str,
                       retry: int = 5) -> None:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        url = f'https://api.telegram.org/bot{API_TOKEN}/sendPhoto'

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
                    'caption': msg_text,
                    'reply_markup': ikb
                },
                files={'photo': open(photo_path, 'rb')}
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

        all_prices = self.prices_data[2][-150:] + self.prices_data[3] + self.prices_data[4]
        max_price = max(all_prices)
        min_price = min(all_prices)

        for price_index in (2, 3, 4):
            new_prices = []

            for price in self.prices_data[price_index][-150:]:
                new_price = (price - min_price) / (max_price - min_price)
                new_prices.append(float("%.10f" % new_price))

            data.append(new_prices)

        return data
    


    def __check_neural_network(self, normalizated_data: list) -> str:
        model = keras.models.load_model(f'saved_models/{self.current_position}.keras')
        result = model.predict(np.array([normalizated_data]))

        if result[0][0] > result[0][1]:
            action = 'BUY'
        else:
            action = 'NOT_BUY'
        logging.info(f'nn decision => {action}')

        return action
    


    def __create_photo(self,
                       normalizated_data: list[float],
                       candle_time: str) -> str:
        draw_graph = DrawGraph(normalizated_data)
        filename = f'{self.symbol}_{self.interval}_{candle_time}.png'
        filename = filename.replace(' ', '_').replace('/', '_').replace(':', '_')
        savepath = os.path.join(ROOT_DIR, CANDLE_IMAGE_DIR, filename)
        draw_graph.show_graph_line(save=True, savepath=savepath)

        return savepath



    def create_request(candles):
        def decorator(func):
            def wrapper(self):
                url = f"https://api-adapter.dzengi.com/api/v1/klines?symbol={self.symbol_no_slash}&interval={self.interval}&limit={candles}"
                r = requests.get(url)

                try:
                    r = r.json()
                except Exception as _ex:
                    logging.critical(f'{_ex} ==> {r.status_code} <=> {r}')
                result = func(self, response=r)
                return result
            return wrapper
        return decorator
    


    @create_request(candles=300)
    def get_prices(self, **kwargs) -> list[list[float]]:
        # get the 3 lists with the last several hundreds
        # high prices, low prices, close prices
        response = kwargs['response']
        self.last_kline_start_time = str(response[-1][0])
        data = []
        for i in (2, 3):    # high price, low price
            item = [float(kline[i]) for kline in response[150:]]
            data.append(item)
        
        data.append([float(kline[4]) for kline in response])   # close prices

        ema_50, ema_150 = self.__get_start_ema(data[-1])
        data.append(ema_50)
        data.append(ema_150)

        return data   # prices
    


    @create_request(candles=2)
    def get_last_prices_for_update(self, **kwargs) -> list[float]:
        # get prices of the last closed candle
        return kwargs['response']
        


    def place_order(self, order_type: str) -> None:     # buy or sell
        if order_type.lower() == 'buy':
            order = self.client.create_order(symbol=SYMBOL, side='buy', type='MARKET')
            print('Open position', order)
        elif order_type.lower() == 'sell':
            order = self.client.create_order(symbol=SYMBOL, side='sell', type='MARKET')
            print('Close position', order)



    def on_message(self, klines: list) -> None or False:    # get data from websockets        
        # if a new candle has started
        kline_start_time = str(klines[-1][0])

        if kline_start_time == self.last_kline_start_time:
            # delete first price in lists and add new price in the list end
            count = 0
            for i in (2, 3, 4):            # high price, low price, close price
                self.prices_data[count].pop(-1)
                self.prices_data[count].append(float(klines[-1][i]))
                count += 1

            self.__calc_ema()

        elif str(klines[-2][0]) == self.last_kline_start_time:
            # delete last price in lists and add new price in the list end
            # self.candles_count += 1
            # print(f'new_candle => {self.candles_count}')
            count = 0
            for i in (2, 3, 4):            # high price, low price, close price    
                self.prices_data[count].pop(-1)
                self.prices_data[count].append(float(klines[-2][i]))
                self.prices_data[count].pop(0)
                self.prices_data[count].append(float(klines[-1][i]))
                count += 1

            self.__calc_ema()
            self.last_kline_start_time = kline_start_time

            if self.is_intersection:
                self.candles_after_intersection += 1
                if self.candles_after_intersection > 15:
                    self.__reset_signal()
                    if self.buy: self.buy = False
        else:
            return False

        # unix_time = int(self.last_kline_start_time[:-3])
        # candle_start_time = datetime.fromtimestamp(unix_time)
        # normalizated_data = self.__normalization()
        # self.__create_photo(normalizated_data, candle_start_time)
        
        slowk, slowd = indicators.stochastic(
            high=pd.Series(self.prices_data[0]),
            low=pd.Series(self.prices_data[1]),
            close=pd.Series(self.prices_data[2][-150:]),
            period=5, perc_k_smoothing=5, perc_d_smoothing=5
        )

        blue = tuple(slowk)[-2:]
        orange = tuple(slowd)[-2:]

        # print(f"[{round(blue[-2], 2)}] {round(blue[-1], 2)}")
        # print(f"[{round(orange[-2], 2)}] {round(orange[-1], 2)}\n")
        # unix_time = int(self.last_kline_start_time[:-3])
        # candle_start_time = datetime.fromtimestamp(unix_time)
        # print(f"{candle_start_time} <=> EMA 150 => [{self.prices_data[4][-2]}] {self.prices_data[4][-1]}\n")

        if self.__stoch_is_valid(blue, orange) and \
            self.__entry_point_ema_is_valid():

            logging.info(f'Intersecion ema => {self.current_position}')
            normalizated_data = self.__normalization()
            nn_action = self.__check_neural_network(normalizated_data)
            action_by_trend = self.__check_trend()

            if (nn_action == 'BUY' or action_by_trend == 'BUY') and (not self.buy):
                log_text = f'close price => {self.prices_data[2][-1]}\n' \
                    f'ema 50 => {self.prices_data[3][-1]}\n' \
                    f'ema 150 => {self.prices_data[4][-1]}\n' \
                    f'stoch orange => {round(orange[-1], 2)}\n' \
                    f'stoch blue => {round(blue[-1], 2)}\n' \
                    f'NN => {nn_action}\n' \
                    f'TREND => {action_by_trend}\n'
                logging.info(log_text)
                
                self.buy = True

                unix_time = int(self.last_kline_start_time[:-3])
                candle_start_time = datetime.fromtimestamp(unix_time)
                path = self.__create_photo(normalizated_data, candle_start_time)

                all_text = f'❗️❗️❗️СИГНАЛ❗️❗️❗️\n{self.symbol}\n{self.interval}\n' \
                    f'{self.current_position}\nВремя начала свечи: {candle_start_time}\n\n' \
                    f'{log_text}'
                self.__send_message(msg_text=all_text, photo_path=path)



    def streaming(self):
        while True:
            klines = self.get_last_prices_for_update()
            if self.on_message(klines) == False: return

            time.sleep(50)
                 

    def __del__(self):
        return




def main(symbol: str, interval: str) -> None:
    retry = 5

    while True:
        if retry > 0:
            stream = Trade(symbol=symbol.upper(), interval=interval.lower())
            stream.streaming()
        else:
            logging.critical('График конкретно пошел по пизде')
            raise Exception('График конкретно пошел по пизде')
        
        logging.error('Похоже график пошел по пизде')
        retry -= 1



if __name__ == '__main__':
    symbol = input('Ввеите монету (Пример: BTC/USD) => ')
    interval = input('Введите таймфрейм (Примеры: 1m, 5m, 15m) => ')

    logging.basicConfig(
        level=logging.INFO,
        filename=f"log_test_{symbol.replace('/', '_')}_{interval}.log",
        filemode='w',
        format="%(asctime)s %(levelname)s %(message)s"
    )

    image_dir_path = os.path.join(ROOT_DIR, CANDLE_IMAGE_DIR)
    if not os.path.exists(image_dir_path):
        os.mkdir(image_dir_path)

    main(symbol, interval)

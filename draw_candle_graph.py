import os
import json

import pandas as pd
import matplotlib.pyplot as plt



class DrawGraph:
    """Draw candlestick graph with EMA"""


    def __init__(self,
                 data: list[list[float, int]] = None,
                 filepath: str = None,
                 directory: str = None) -> None:
        self.full_data = None
        
        if data:
            self.full_data = {
                'close': data[0],
                'ema_50': data[1],
                'ema_150': data[2]
            }
        elif filepath:
            self.full_data = self.__get_data_from_json(filepath)
        elif directory:
            self.directory = directory



    def show_graph_bar(self,
                   save: bool = False,
                   savepath: str = None) -> None:
        # DataFrame to represent opening , closing, high 
        # and low prices of a stock for a week
        stock_prices = pd.DataFrame(self.full_data)
        
        plt.figure()
        
        # "up" dataframe will store the stock_prices 
        # when the closing stock price is greater
        # than or equal to the opening stock prices
        up = stock_prices[stock_prices.close >= stock_prices.open]
        
        # "down" dataframe will store the stock_prices
        # when the closing stock price is
        # lesser than the opening stock prices
        down = stock_prices[stock_prices.close < stock_prices.open]
        
        # When the stock prices have decreased, then it
        # will be represented by blue color candlestick
        col1 = 'green'
        
        # When the stock prices have increased, then it 
        # will be represented by green color candlestick
        col2 = 'red'
        
        # Setting width of candlestick elements
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

        col_ema_50 = 'blue'
        col_ema_150 = 'olive'
        
        plt.plot(stock_prices.ema_50, color=col_ema_50)
        plt.plot(stock_prices.ema_150, color=col_ema_150)

        if save:
            plt.savefig(savepath)
        else:
            # displaying candlestick chart of stock data 
            # of a week
            plt.show()



    def show_graph_line(self,
                        save: bool = False,
                        savepath: str = None) -> None:
        # DataFrame to represent opening , closing, high 
        # and low prices of a stock for a week
        stock_prices = pd.DataFrame(self.full_data)
        
        plt.figure()
        
        col_price = 'red'
        col_ema_50 = 'blue'
        col_ema_150 = 'olive'
        
        plt.plot(stock_prices.close, color=col_price, linewidth=0.8)
        plt.plot(stock_prices.ema_50, color=col_ema_50, linewidth=0.8)
        plt.plot(stock_prices.ema_150, color=col_ema_150, linewidth=0.8)

        if save:
            plt.savefig(savepath)
        else:
            # displaying candlestick chart of stock data 
            # of a week
            plt.show()



    def __get_data_from_json(self, filepath) -> dict[list[float]]:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        return data
    
    

    def view_all_by_directory(self):
        directory_list = os.listdir(self.directory)
        count = 0

        for file in directory_list:
            count += 1
            self.full_data = self.__get_data_from_json(fr"{self.directory}\{file}")
            print(f'{count} - {file}')
            self.show_graph()
            input('Press Enter to next')




if __name__ == '__main__':
    print('Привет, я нарисую тебе график\n')
    filepath = r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\test\negative_short_5mUNFIUSDT2023-08-27_09-24-59_binance.json'
    graph = DrawGraph(filepath)
    
    input('Нажмите Enter чтобы выйти')
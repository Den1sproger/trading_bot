import os
import json

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, SimpleRNN, Input, LSTM, GRU, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad



# SHORT_SAMPLE = r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\short'
# LONG_SAMPLE = r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\long'
# NEGATIVE_SAMPLE = r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\negative_signals'



class Training_sample:
    """"""

    SAMPLE_DIRECTORIES = {
        'short': r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\short',
        'long': r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\long',
        'negative': r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\negative'
    }

    TARGET_BUY = np.array([1, 0])
    TARGET_NOT_BUY = np.array([0, 1])


    def __init__(self, position: str = None) -> None:
        if position: assert position in ('short', 'long'), 'Unknown position'
        self.position = position
        self.max_price: float
        self.min_price: float
        

    def _create_borders(self, signal_data: dict[list[float]]) -> None:
        all_prices = np.append(signal_data['open'],
                               [signal_data['close'],
                                signal_data['ema_50'],
                                signal_data['ema_150']])
        self.max_price = max(all_prices)
        self.min_price = min(all_prices)


    def _get_normal_price(self,
                           price: float) -> float:
        new_price = (price - self.min_price) / (self.max_price - self.min_price)
        return float("%.10f" % new_price)


    def _normalization(self, prices: np.ndarray) -> np.ndarray:
        new_prices = np.array([])

        for price in prices:
            new_price = self._get_normal_price(price)
            new_prices = np.append(new_prices, new_price)

        return new_prices
        

    def get_data_from_training_sample(self) -> tuple[np.ndarray]:
        input_data_train = []
        output_data_train = []
        input_data_valid = []
        output_data_valid = []

        for key, path in self.SAMPLE_DIRECTORIES.items():
            directory = os.listdir(path)

            input_data = []
            output_data = []

            for item in directory:
                with open(fr'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\{key}\{item}', 'r', encoding='utf-8') as file:
                    data = json.load(file)

                self._create_borders(signal_data=data)
                upgrade_data = []

                for i in ['close', 'ema_50', 'ema_150']:
                    normalizated_prices = self._normalization(np.array(data[i]))
                    upgrade_data.append(normalizated_prices)

                input_data.append(np.array(upgrade_data))

            if key == self.position:
                target = self.TARGET_BUY
                size_val = 5

            else:
                target = self.TARGET_NOT_BUY
                if key == 'negative':
                    size_val = 7
                else:
                    size_val = 16

            input_data_train.append(input_data[size_val:])
            input_data_valid.append(input_data[:size_val])

            output_data = [target for _ in range(len(input_data))]
            output_data_train.append(output_data[size_val:])
            output_data_valid.append(output_data[:size_val])

        input_data_train = [j for i in input_data_train for j in i]
        output_data_train = [j for i in output_data_train for j in i]
        input_data_valid = [j for i in input_data_valid for j in i]
        output_data_valid = [j for i in output_data_valid for j in i]

        return np.array(input_data_train), np.array(output_data_train), np.array(input_data_valid), np.array(output_data_valid)
    

    def __del__(self) -> None:
        return
    


class Testing_sample(Training_sample):
    """"""

    SAMPLE_DIRECTORY = r'C:\Users\Денис\vscode_projects\binance_bot\neural_networks\training_sample\test'

    # def get_normalize_prices(self, data: dict[list[float]]) -> np.ndarray:
    #     self._create_borders(signal_data=data)
    #     upgrade_data = []

    #     for i in ['open', 'close', 'ema_50', 'ema_150']:
    #         normalizated_prices = self._normalization(np.array(data[i]))
    #         upgrade_data.append(normalizated_prices)

    #     return np.array(upgrade_data)
    

    def get_data_from_testing_sample(self) -> list[np.ndarray]:
        directory = os.listdir(self.SAMPLE_DIRECTORY)

        input_data = []

        for item in directory:
            with open(fr'{self.SAMPLE_DIRECTORY}\{item}', 'r', encoding='utf-8') as file:
                data = json.load(file)

            self._create_borders(signal_data=data)
            upgrade_data = []

            for i in ['close', 'ema_50', 'ema_150']:
                normalizated_prices = self._normalization(np.array(data[i]))
                upgrade_data.append(normalizated_prices)

            input_data.append(np.array(upgrade_data))

        return directory, input_data
    


def main():
    ts = Training_sample(position='short')
    x_train_data, y_train_data, x_val_data, y_val_data = ts.get_data_from_training_sample()
    # with open('result_train.json', 'w', encoding='utf-8') as file:
    #     json.dump(x_train_data, file, indent=4, ensure_ascii=False)
    # print(len(x_train_data))
    # print(len(y_train_data[0]))
    # print(len(x_train_data))
    # print(len(y_train_data))
    # print(len(x_val_data))
    # print(len(y_val_data))

    # create the network model
    model = Sequential()
    # model.add(Input(150, 4))
    model.add(Dense(4, activation='relu', input_shape=(3, 150)))
    model.add(Dense(23, activation='relu'))
    # model.add(LSTM(units=20))
    model.add(GRU(35))
    # model.add(Dense(25, activation='relu'))
    # model.add(GRU(20))
    # model.add(Dropout(0.7))
    # model.add(GRU(15))
    # model.add(LSTM(units=32))
    # model.add(Dense(units=7, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    model.summary()

    # sgd_optimzer = SGD(nesterov=True)
    # adagard = Adagrad()
    
    # compile the network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train the model
    history = model.fit(x_train_data, y_train_data, epochs=70, validation_data=(x_val_data, y_val_data))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid(True)
    plt.show()

    test_sample = Testing_sample()
    filenames, test_data = test_sample.get_data_from_testing_sample()
    for file, item in zip(filenames, test_data):
        # print(file)
        # print(item)
        print(f"{file} => {model.predict(np.array([item]))}")



if __name__ == '__main__':
    main()
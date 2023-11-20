import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, SimpleRNN, Input, LSTM, GRU, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD, Adagrad



class Training_sample:
    """"""

    SAMPLE_DIRECTORIES = {
        # 'short_big': [r'C:\Users\Денис\vscode_projects\trading_bot\term_short\short_big', 8],
        # 'short_small': [r'C:\Users\Денис\vscode_projects\trading_bot\term_short\short_small', 14],
        # 'short_universe': [r'C:\Users\Денис\vscode_projects\trading_bot\term_short\short_universe', 4],
        # 'long_big': [r'C:\Users\Денис\vscode_projects\trading_bot\term_long\long_big', 8],
        # 'long_small': [r'C:\Users\Денис\vscode_projects\trading_bot\term_long\long_small', 14],
        # 'long_universe': [r'C:\Users\Денис\vscode_projects\trading_bot\term_long\long_universe', 4],
        'short_strong': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\short_strong', 8],
        'short_middle':[r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\short_middle', 7],
        'short_weak': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\short_weak', 5],
        # 'short_5_strong': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\short_5_candles\short_5_strong', 3],
        # 'short_5_middle':[r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\short_5_candles\short_5_middle', 2],
        # 'short_5_weak': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\short_5_candles\short_5_weak', 3],
        'long_strong': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\long_strong', 8],
        'long_middle': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\long_middle', 7],
        'long_weak': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\long_weak', 5],
        # 'long_5_strong': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\long_5_candles\long_5_strong', 3],
        # 'long_5_middle': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\long_5_candles\long_5_middle', 2],
        # 'long_5_weak': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\long_5_candles\long_5_weak', 3],
        'negative': [r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\negative', 30]
    }

    TARGET_BUY = np.array([1, 0])
    TARGET_NOT_BUY = np.array([0, 1])


    def __init__(self, position: str = None) -> None:
        if position: assert position in ('short', 'long'), 'Unknown position'
        self.position = position
        

    def _normalization(self, signal_data: list[list[float]]) -> list[any]:
        data = []

        all_prices = signal_data['close'] + signal_data['ema_50'] + signal_data['ema_150']
        max_price = max(all_prices)
        min_price = min(all_prices)

        for price_type in ('close', 'ema_50', 'ema_150'):
            new_prices = []

            for price in signal_data[price_type]:
                new_price = (price - min_price) / (max_price - min_price)
                new_prices.append(float("%.10f" % new_price))

            data.append(new_prices)

        return data
    

    def _split_on_train_valid(self, train_data: list, size: int) -> list[any]:
        valid_data = []

        for _ in range(size):
            random_index = random.choice(range(len(train_data)))
            valid_data.append(train_data[random_index])
            train_data.pop(random_index)

        return train_data, valid_data
        

    def get_data_from_training_sample(self) -> tuple[np.ndarray]:
        input_data_train = []
        output_data_train = []
        input_data_valid = []
        output_data_valid = []

        for key, path in self.SAMPLE_DIRECTORIES.items():
            directory = os.listdir(path[0])

            input_data = []
            output_data = []

            for item in directory:
                with open(fr'{path[0]}\{item}', 'r', encoding='utf-8') as file:
                    data = json.load(file)

                upgrade_data = self._normalization(data)
                input_data.append(upgrade_data)

            if self.position in key:
                target = self.TARGET_BUY
            else:
                target = self.TARGET_NOT_BUY

            size_val = self.SAMPLE_DIRECTORIES[key][1]

            output_data = [target for _ in range(len(input_data))]
            output_data_train.append(output_data[size_val:])
            output_data_valid.append(output_data[:size_val])

            current_input_train, current_input_valid = self._split_on_train_valid(
                train_data=input_data, size=size_val
            )
            input_data_train.append(current_input_train)
            input_data_valid.append(current_input_valid)

        input_data_train = [j for i in input_data_train for j in i]
        output_data_train = [j for i in output_data_train for j in i]
        input_data_valid = [j for i in input_data_valid for j in i]
        output_data_valid = [j for i in output_data_valid for j in i]

        return np.array(input_data_train), np.array(output_data_train), np.array(input_data_valid), np.array(output_data_valid)
    

    def __del__(self) -> None:
        return
    


class Testing_sample(Training_sample):
    """"""

    SAMPLE_DIRECTORY = r'C:\Users\Денис\vscode_projects\trading_bot\neural_networks\training_sample\test'
    

    def get_data_from_testing_sample(self) -> list[np.ndarray]:
        directory = os.listdir(self.SAMPLE_DIRECTORY)

        input_data = []

        for item in directory:
            with open(fr'{self.SAMPLE_DIRECTORY}\{item}', 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            upgrade_data = self._normalization(data)
            input_data.append(np.array(upgrade_data))

        return directory, input_data
    


def main():
    position = 'short'
    # ts = Training_sample(position)
    # x_train_data, y_train_data, x_val_data, y_val_data = ts.get_data_from_training_sample()

    # # create the network model
    # model = Sequential()
    # model.add(Dense(3, activation='relu', input_shape=(3, 150)))
    # model.add(Dense(10, activation='relu'))
    # model.add(GRU(30))
    # model.add(Dense(units=2, activation='softmax'))
    # model.summary()
    
    # # compile the network
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # # train the model
    # history = model.fit(x_train_data, y_train_data, epochs=100, validation_data=(x_val_data, y_val_data))

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.grid(True)
    # plt.show()

    test_sample = Testing_sample()
    filenames, test_data = test_sample.get_data_from_testing_sample()

    model = load_model('saved_models/short_1.keras')

    for file, item in zip(filenames, test_data):
        result = model.predict(np.array([item]))
        if result[0][0] > result[0][1]:
            action = 'BUY'
        else:
            action = 'NOT_BUY'
        print(f"{file} => {result} <=> {action}")

    # model.save(f'saved_models/{position}.keras')
    


if __name__ == '__main__':
    main()
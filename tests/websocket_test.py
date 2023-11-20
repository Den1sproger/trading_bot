import json
import time

import websocket


def on_message(ws, message):
    data = json.loads(message)
    # last_price = data.get("k").get("c")
    print(data)
    time.sleep(2)


def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    print("### connected ###")


def main():
    url = r'wss://api-adapter.dzengi.com/connect/api/v2/klines?interval="1m"&symbol="BTC%2FUSD"'
    wsa = websocket.WebSocketApp(
        url, on_message=on_message,
        on_error = on_error
        # on_close = on_close
    )
    wsa.on_open = on_open
    wsa.run_forever()


if __name__ == '__main__':
    main()

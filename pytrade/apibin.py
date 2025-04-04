import pandas as pd
from binance.spot import Spot

# Place an order on Binance
def place_order(client, side, symbol, quantity):
    try:
        order = client.new_order(
            symbol=symbol,
            side=side.upper(),
            type="MARKET",
            quantity=quantity
        )
        return order
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

# Fetch historical data from Binance
def fetch_klines(client, symbol, interval, limit=100):
    try:
        klines = client.klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "time", "open", "high", "low", "close", "volume", "close_time", 
            "qav", "num_trades", "taker_base", "taker_quote", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return None

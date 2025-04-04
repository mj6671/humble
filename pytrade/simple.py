import ccxt
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
import os
from models.model import fetch_pending_orders, fetch_user_api_keys, save_order_to_db, update_users_orders,log_trade,update_order_status
from datetime import datetime
from config.database import db_connect
from decimal import Decimal, ROUND_DOWN
import math

load_dotenv()

RSI_PERIOD = int(os.getenv("RSI_PERIOD"))
RSI_LENGTH = int(os.getenv("RSI_LENGTH"))
RSI_UPPER = int(os.getenv("RSI_UPPER"))
RSI_MIDDLE = int(os.getenv("RSI_MIDDLE"))
RSI_LOWER = int(os.getenv("RSI_LOWER"))
TIMEFRAME = os.getenv("TIMEFRAME")

BINANCE_TESTNET_API = os.getenv("BINANCE_TESTNET_API")
BINANCE_TESTNET_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
# Binance API setup
binance = ccxt.binance({
    'apiKey': BINANCE_TESTNET_API,
    'secret': BINANCE_TESTNET_SECRET,
    'options': {'defaultType': 'spot'},
})
# Connect to Testnet Endpoint
binance.set_sandbox_mode(True)  # Enables testnet mode
# Function to fetch OHLCV data
def fetch_ohlcv(symbol):
    bars = binance.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    #print(f"red {df}")  # Proper indentation
    
    return df

# Function to convert to Heikin-Ashi candles
def convert_to_heikin_ashi(df):
    ha_df = df.copy()
    
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha_df['high'] = df[['high', 'open', 'close']].max(axis=1)
    ha_df['low'] = df[['low', 'open', 'close']].min(axis=1)

    ha_df.dropna(inplace=True)  # Remove first row due to NaN from shift
    return ha_df

# Function to calculate RSI
def calculate_rsi(data, period=RSI_PERIOD):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_min_trade_amount(symbol):
    market = binance.market(symbol)
    return float(market['limits']['amount']['min'])
	
def get_step_size(symbol):
    market = binance.market(symbol)
    return float(market['precision']['amount'])
	
def get_min_notional(symbol):
    market = binance.market(symbol)
    return float(market['limits']['cost']['min'])
	
def get_latest_price(symbol):
    ticker = binance.fetch_ticker(symbol)  
    last_price = ticker['last']
    if last_price is None:
        raise ValueError(f"Error: Could not fetch last price for {symbol}")
    return last_price  # Ensure return is properly aligned


# Simple Buy Function
def simple_buy(user_id, order_id, exchange_id, symbol, amount):
    df = fetch_ohlcv(symbol)
    ha_df = convert_to_heikin_ashi(df)
    ha_df["rsi"] = calculate_rsi(ha_df)

    last_candle = ha_df.iloc[-2]
    current_candle = ha_df.iloc[-1]
	
	

    if current_candle["ha_close"] > last_candle["ha_close"] and ha_df["rsi"].iloc[-1] < RSI_LOWER:
        exchange_order_id=0
        try:
            order = binance.create_market_buy_order(symbol, amount)
            price = order["fills"][0]["price"] if order.get("fills") else order["price"]
            exchange_order_id= order["id"] 
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, "Simple Buy", price, amount, "executed")
            update_order_status(user_id, order_id, "COMPLETE")
            print(f"BUY ORDER EXECUTED for User {user_id}: {order}")

        except Exception as e:
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, "Simple Buy", 0, amount, "failed")
            print(f"BUY ORDER FAILED for User {user_id}: {e}")
    else:
        print(f"No Buy Signal for User {user_id} for {symbol}")

# Simple Sell Trade Function using Heikin-Ashi
def simple_sell(user_id, order_id, exchange_id, symbol, amount):
    df = fetch_ohlcv(symbol)
    ha_df = convert_to_heikin_ashi(df)
    ha_df["rsi"] = calculate_rsi(ha_df)

    last_candle = ha_df.iloc[-2]
    current_candle = ha_df.iloc[-1]

    min_notional = get_min_notional(symbol)  # Get min trade value in USDT
    min_amount = get_min_trade_amount(symbol)  # Get min amount for the pair
    step_size = get_step_size(symbol)  # Get correct precision
    # Fetch latest price to calculate notional value
    
    last_price = get_latest_price(symbol)
    min_notional = Decimal(str(min_notional))  # Convert min_notional to Decimal
    step_size = Decimal(str(step_size))
    amount = Decimal(amount)
    last_price = Decimal(str(last_price))  # Convert last_price to Decimal
    order_value = amount * last_price  # Multiply correctly

    # Adjust if below min_notional
    if order_value < min_notional:
        adjusted_amount = min_notional / last_price
        adjusted_amount = math.floor(adjusted_amount / step_size) * step_size  # Round to step size
       # print(f"Adjusting amount from {amount} to {adjusted_amount} to meet min notional.")
        amount = adjusted_amount

    if amount < min_amount:
        print(f"Final amount {amount} is still below min trade size. Order skipped.")
        return

    if current_candle["ha_close"] < last_candle["ha_close"] and ha_df["rsi"].iloc[-1] > RSI_UPPER:
        exchange_order_id=0
        try:
            order = binance.create_market_sell_order(symbol, amount)
            price = order["fills"][0]["price"] if order.get("fills") else order["price"]
            exchange_order_id = order["id"]
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, "Simple Sell", price, amount, "executed")
            update_order_status(user_id, order_id, "Complete")
           # print(f"exchange_order_id: {exchange_order_id}")
            print(f"SELL ORDER EXECUTED for User {user_id}: {order}")

        except Exception as e:
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, "Simple Sell", 0, amount, "failed")
            print(f"SELL ORDER FAILED for User {user_id}: {e}")
    else:
        print(f"No Sell Signal for User {user_id} for {symbol}")


def process_pending_orders():
    while True:
        conn = db_connect()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM trades WHERE status = 'Pending'")
        orders = cursor.fetchall()
        conn.close()

        for order in orders:
            user_id = order["user_id"]
            order_id = order["id"]
            exchange_id = order["exchange_id"]
            symbol = order["currency"]
            order_type = order["type"]
            amount = order["qty"]

            if order_type == "Simple Buy":
                simple_buy(user_id, order_id, exchange_id, symbol, amount)
            elif order_type == "Simple Sell":
                simple_sell(user_id, order_id, exchange_id, symbol, amount)

        time.sleep(3)
# Example Usage
if __name__ == "__main__":
    process_pending_orders()
   #fetch_ohlcv("ETH/USDT")

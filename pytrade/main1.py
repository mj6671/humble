import time
import os
from dotenv import load_dotenv
from config.database import db_connect 
from models.model import fetch_pending_orders, fetch_user_api_keys, save_order_to_db, update_users_orders, get_exchange_details
from strategies.strategies import calculate_rsi, calculate_supertrend, check_entry_conditions
from services.exchanges.exchange_factory import get_exchange_client
import pandas as pd

def calculate_variable_stop_loss(order_price, percentage):
    return order_price * (1 - percentage / 100)

def get_last_bearish_candle_low(df):
    bearish_candles = df[df['close'] < df['open']]
    return bearish_candles.iloc[-1]['low'] if not bearish_candles.empty else None

def get_last_bullish_candle_high(df):
    bullish_candles = df[df['close'] > df['open']]
    return bullish_candles.iloc[-1]['high'] if not bullish_candles.empty else None

def get_low_of_last_n_candles(df, n):
    return df['low'].iloc[-n:].min()

def get_high_of_last_n_candles(df, n):
    return df['high'].iloc[-n:].max()

def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha_df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
    ha_df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
    return ha_df

def calculate_stop_loss(order_price, df, percentage, trade_type):
    variable_sl = calculate_variable_stop_loss(order_price, percentage)
    
    if trade_type == 'BUY':
        sl_candidates = [
            get_last_bearish_candle_low(df),
            get_low_of_last_n_candles(df, 3),
            get_low_of_last_n_candles(df, 2)
        ]
    elif trade_type == 'SELL':
        sl_candidates = [
            get_last_bullish_candle_high(df),
            get_high_of_last_n_candles(df, 3),
            get_high_of_last_n_candles(df, 2)
        ]
    
    sl_candidates = [sl for sl in sl_candidates if sl is not None]
    
    if sl_candidates:
        calculated_sl = min(sl_candidates) if trade_type == 'buy' else max(sl_candidates)
        return max(variable_sl, calculated_sl) if trade_type == 'buy' else min(variable_sl, calculated_sl)
    else:
        return variable_sl

load_dotenv()

RSI_PERIOD = int(os.getenv("RSI_PERIOD"))
SUPER_TREND_LENGTH = int(os.getenv("SUPER_TREND_LENGTH"))
SUPER_TREND_MULTIPLIER = int(os.getenv("SUPER_TREND_MULTIPLIER"))
TIMEFRAME = os.getenv("TIMEFRAME")
STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE"))

def main():
    while True:
        try:
            print("\nChecking for pending orders...")
            orders = fetch_pending_orders()
            print(f"Orders found: {len(orders)}")
            
            for order in orders:
                user_id = order["user_id"]
                exchange_id = order["exchange_id"]
                exchange_details = get_exchange_details(exchange_id)

                if not exchange_details:
                    print(f"❌ No exchange details found for exchange_id {exchange_id}")
                    continue

                print(f"\nProcessing order for user {user_id} on exchange_id - {exchange_id} - {order}")
                api_keys = fetch_user_api_keys(user_id, exchange_id)
                if not api_keys:
                    print(f"❌ No API keys found for user {user_id}")
                    continue

                exchange_client = get_exchange_client(exchange_details['exchange_name'], api_keys["api_key"], api_keys["secret_key"])
                df = exchange_client.retrieve_candlestick_data(order["currency_symbol"], TIMEFRAME, limit=50)
                
                if df is None or df.empty:
                    print(f"⚠️ No market data available for {order['currency_symbol']}")
                    continue

                df = calculate_rsi(df, RSI_PERIOD)
                df = calculate_supertrend(df, SUPER_TREND_LENGTH, SUPER_TREND_MULTIPLIER)
                df = calculate_heikin_ashi(df)
                action, conditions_json = check_entry_conditions(df)
                print(f"📊 Strategy Decision: {action}")

                if action == order["order_type"]:
                    print(f"✅ Placing {action} order for {order['currency_symbol']} on {exchange_details['exchange_name']}...")
                    order_response = exchange_client.create_market_order(action, order["currency_symbol"], order["quantity"])
                    
                    if order_response:
                        stop_loss = calculate_stop_loss(float(order_response['fills'][0]['price']), df, STOP_LOSS_PERCENTAGE, action)
                        print(f"🚨 Calculated Stop Loss for {order['currency_symbol']}: {stop_loss}")
                        
                        save_order_to_db(order_response, user_id, order["order_id"], exchange_id, conditions_json)
                        update_users_orders(order_response, user_id, order["order_id"], exchange_id)
                    else:
                        print("⚠️ Order placement failed!")
        except Exception as e:
            print(f"❌ Error occurred: {e}")
        
        print("\n⏳ Waiting 1 second before next check...\n")
        time.sleep(1)

if __name__ == "__main__": 
    main()

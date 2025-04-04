import time
import os
from dotenv import load_dotenv
from decimal import Decimal, ROUND_DOWN
#from services.orders.orderPlacement import orderPlacement
#from services.orders.orderConditions import orderConditions
from strategies.strategies import calculate_rsi, calculate_supertrend, check_entry_conditions, calculate_rsi2
from models.model import fetch_pending_orders, fetch_user_api_keys, save_order_to_db, update_users_orders, log_trade,update_order_status, update_trailing_stoploss
from services.exchanges.exchange_factory import ExchangeFactory
from helpers.commonHelper import commonHelper
load_dotenv()



def main():
    while True:
        try:
            rsi_setting = commonHelper.get_setting('rsi')
            supertrend_setting = commonHelper.get_setting('supertrend')
            RSI_PERIOD = int(rsi_setting.get('period'))
            RSI_UPPER_LIMIT = int(rsi_setting.get('upper_limit'))
            RSI_MIDDLE_LIMIT = int(rsi_setting.get('middle_limit'))
            RSI_LOWER_LIMIT = int(rsi_setting.get('lower_limit'))
            SUPER_TREND_LENGTH = int(supertrend_setting.get('length'))
            SUPER_TREND_MULTIPLIER = int(supertrend_setting.get('factor'))
            TIMEFRAME = os.getenv("TIMEFRAME")
            STOP_LOSS_PERCENTAGE = Decimal(os.getenv("STOP_LOSS_PERCENTAGE"))
            
            orders = fetch_pending_orders() 
            if not orders:
                print("No pending orders. Waiting for new orders...")
                
            for order in orders:
                
                order_type = order['type']
                user_id = order["user_id"]
                exchange_id = str(order["exchange_id"])  # Ensure exchange_id is a string
                currency_symbol = order["currency"]
                order_price = Decimal(order["buy_rate"])
                
                order_type2 = 'BUY'
                
                if order_type in ['Simple Buy', 'Smart Buy']:
                    order_type2 = 'BUY'
                elif order_type in ['Simple Sell', 'Smart Sell']:
                    order_type2 = 'SELL'
                
                print(f"\nProcessing order for user {user_id} on exchange_id {exchange_id} - {order}")
                
                # Fetch user's API keys based on exchange_id
                api_keys = fetch_user_api_keys(user_id, exchange_id)
                if not api_keys:
                    print(f"❌ No API keys found for user {user_id} on exchange_id {exchange_id}")
                    continue

                exchange_client = ExchangeFactory.get_exchange_client(exchange_id, api_keys["api_key"], api_keys["secret_key"])
                
                candle_limit= os.getenv("CANDLES_LIMIT")
                # Fetch market data
                df = exchange_client.retrieve_candlestick_data(currency_symbol, TIMEFRAME, candle_limit)
                
                if df is None or df.empty:
                    print(f"⚠️ No market data available for {currency_symbol} on exchange_id {exchange_id}")
                    continue

                # Apply strategies
                df = calculate_rsi(df, RSI_PERIOD)
                df = calculate_supertrend(df, SUPER_TREND_LENGTH, SUPER_TREND_MULTIPLIER)
                
                commonHelper.check_update_stoploss(order, order_price, STOP_LOSS_PERCENTAGE, df, RSI_LOWER_LIMIT, RSI_UPPER_LIMIT, order_type2)
                
                #if order_type == 'Simple Buy':
                    #order_data = orderConditions.simple_buy_conditions(order)
                    #if order_data:
                        #orderPlacement.simple_buy(order_data)
                
                
        except Exception as e:
            import traceback
            print(f"❌ Error occurred: {e}")
            print(traceback.format_exc())
        
        print("\n⏳ Waiting 1 second before next check...\n")
        time.sleep(3)

if __name__ == "__main__": 
    main()

import time
import os
from dotenv import load_dotenv
from config.database import db_connect 
from models.model import fetch_pending_orders, fetch_user_api_keys, save_order_to_db, update_users_orders, get_exchange_details
from strategies.strategies import calculate_rsi, calculate_supertrend, check_entry_conditions, calculate_stop_loss
from services.exchanges.exchange_factory import ExchangeFactory

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
                exchange_id = str(order["exchange_id"])  # Ensure exchange_id is a string
                
                print(f"\nProcessing order for user {user_id} on exchange_id {exchange_id} - {order}")
                
                # Fetch user's API keys based on exchange_id
                api_keys = fetch_user_api_keys(user_id, exchange_id)
                if not api_keys:
                    print(f"❌ No API keys found for user {user_id} on exchange_id {exchange_id}")
                    continue

                exchange_client = ExchangeFactory.get_exchange_client("1", api_keys["api_key"], api_keys["secret_key"])

                # Fetch market data
                df = exchange_client.retrieve_candlestick_data(order["currency_symbol"], TIMEFRAME, limit=50)
                if df is None or df.empty:
                    print(f"⚠️ No market data available for {order['currency_symbol']} on exchange_id {exchange_id}")
                    continue

                # Apply strategies
                df = calculate_rsi(df, RSI_PERIOD)
                df = calculate_supertrend(df, SUPER_TREND_LENGTH, SUPER_TREND_MULTIPLIER)

                action, conditions_json = check_entry_conditions(df)
                print(f"📊 Strategy Decision for {order['currency_symbol']}: {action}")

                if action == order["order_type"]:
                    print(f"✅ Placing {action} order for {order['currency_symbol']} on exchange_id {exchange_id}...")
                    
                    # Place the market order dynamically
                    order_response = exchange_client.create_market_order(
                        action, order["currency_symbol"], order["quantity"]
                    )
                    
                    if order_response:
                        # Calculate stop loss using all conditions
                        order_price = float(order_response['fills'][0]['price'])
                        stop_loss = calculate_stop_loss(order_price, df, STOP_LOSS_PERCENTAGE, action)
                        print(f"🚨 Calculated Stop Loss for {order['currency_symbol']}: {stop_loss}")
                        
                        save_order_to_db(order_response, user_id, order["order_id"], exchange_id, conditions_json)
                        update_users_orders(order_response, user_id, order["order_id"], exchange_id)
                    else:
                        print(f"⚠️ Order placement failed for {order['currency_symbol']} on exchange_id {exchange_id}")
        except Exception as e:
            print(f"❌ Error occurred: {e}")
        
        print("\n⏳ Waiting 1 second before next check...\n")
        time.sleep(1)

if __name__ == "__main__": 
    main()

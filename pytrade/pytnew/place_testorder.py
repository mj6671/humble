import random
import time
from dotenv import load_dotenv
from decimal import Decimal, ROUND_DOWN
import os
from datetime import datetime
from config.database import db_connect  # Ensure `database.py` has a `db_connect` function
from binance.spot import Spot

# Load environment variables
load_dotenv()

STOP_LOSS_PERCENTAGE = Decimal(os.getenv("STOP_LOSS_PERCENTAGE"))
BINANCE_TESTNET_API = os.getenv("BINANCE_TESTNET_API")
BINANCE_TESTNET_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
BASE_URL = "https://testnet.binance.vision"

# Binance client
client = Spot(BINANCE_TESTNET_API, BINANCE_TESTNET_SECRET, base_url=BASE_URL)

def get_currency_price(symbol):
    try:
        # Fetch the current price
        price = client.ticker_price(symbol=symbol)
        
        # Return the price as a float
        return Decimal(price['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

# Currency pairs list
CURRENCY_PAIRS = [
    "ETHUSDT", "BNBUSDT"
]

# Generate test orders
def generate_test_orders():
    connection = db_connect()
    cursor = connection.cursor()

    try:
        # Generate 10 random test orders
        for _ in range(2):
            user_id = 26
            exchange_id = 1
            currency_pair = random.choice(CURRENCY_PAIRS)
            order_type = "Simple Buy"
            price = get_currency_price(currency_pair)
            percentage = STOP_LOSS_PERCENTAGE.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

            if price is None:
                print(f"Skipping order for {currency_pair} due to missing price.")
                continue

            quantity = Decimal("0.01")  # Convert quantity to Decimal
            total = (price * quantity).quantize(Decimal("0.01"), rounding=ROUND_DOWN)  # Total = price * quantity
            
            status = "Pending"  # Default status for test orders
            
            if order_type == "Simple Buy":
                initial_stoploss = (price * (Decimal('1') - percentage / Decimal('100'))).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
                trailing_stoploss = initial_stoploss
            elif order_type == "Simple Sell":
                initial_stoploss = (price * (Decimal('1') + percentage / Decimal('100'))).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
                trailing_stoploss = initial_stoploss

            # Insert test order into users_orders table
            query = """
            INSERT INTO `trades`(`exchange_id`,`user_id`, `currency`, `buy_rate`, `qty`,`total_btc`, `initial_stoploss`, `trailing_stoploss`,`status`, `type`)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (exchange_id, user_id, currency_pair, price, quantity, total, initial_stoploss, trailing_stoploss, status, order_type)
            cursor.execute(query, values)
            print(f"Inserted test order: {values}")

        # Commit the changes
        connection.commit()
        print("Test orders successfully inserted into the database!")

    except Exception as e:
        print(f"Error inserting test orders: {e}")
        connection.rollback()

    finally:
        cursor.close()
        connection.close()

# Fetch supported currency pairs
def fetch_currency_pairs():
    print("Supported Currency Pairs:")
    for pair in CURRENCY_PAIRS:
        print(f"- {pair}")

# Main function to run the script
if __name__ == "__main__":
    print("Welcome to the Test Order Generator!")
    fetch_currency_pairs()
    
    generate_test_orders()
    print("Test order generation completed.")

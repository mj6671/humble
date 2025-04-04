import os
import json
import pandas as pd
import logging

from models.model import fetch_pending_orders, fetch_user_api_keys, save_order_to_db, update_users_orders,log_trade,update_order_status

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class commonHelper:
    """A helper class to retrieve settings from a JSON file."""

    @staticmethod
    def get_setting(key): 
        """Retrieve a specific setting by key from the default settings file."""
        try:
            filelocation = os.path.join(os.getcwd(), "setting.json")
            #logging.info(f"Trying to load settings from: {filelocation}")
            
            with open(filelocation, 'r') as f:
                settings = json.load(f).get("settings", {})
                value = settings.get(key)
                if value is None:
                    logging.warning(f"Key '{key}' not found in setting.json.")
                return value
        except FileNotFoundError:
            logging.error("Settings file not found.")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return None
            
            
   
    @staticmethod
    def get_minimums(symbol):  
        """Retrieve the minimums for a specific trading symbol from the minimums.json file."""
        try:
            # Use the absolute path relative to the current file location
            filelocation = os.path.join(os.getcwd(), "/var/www/html/pytrade/minimums.json")
            with open(filelocation, 'r') as f:
                data = json.load(f)
                # Retrieve minimums for the specified symbol
                minimums = data.get(symbol, None)
                if minimums:
                    return minimums
                else:
                    logging.warning(f"No data found for symbol: {symbol}")
                    return None
        except FileNotFoundError:
            logging.error("Minimums file not found.")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return None
            
    @staticmethod
    def check_update_stoploss(order_data, order_price, percentage, candles, rsi_lower_limit, rsi_upper_limit, order_type):
        try:
            user_id = order_data["user_id"]
            order_id = order_data["id"]
            
            stoploss = calculate_stop_loss(order_price, percentage, candles, rsi_lower_limit, rsi_upper_limit, order_type)
            
            update_trailing_stoploss(user_id, order_id, stoploss)
            print(f"stoploss updated {stoploss}")
            
        except Exception as e:
            import traceback
            logging.error(f"❌ Error occurred: {e}")
            print(traceback.format_exc())
            return None
            
    
    import logging

    def validate_candles(candles):
        """
        Validate the structure of the candles data.
        Ensures it's a list of dictionaries or a DataFrame with required keys/columns.
        
        Args:
            candles (list[dict] | pd.DataFrame): Candles data.

        Raises:
            ValueError: If candles structure is invalid.
        """
        # Check if candles is empty
        if candles is None or (isinstance(candles, pd.DataFrame) and candles.empty):
            raise ValueError("Candles data is empty or None.")
        if isinstance(candles, list) and len(candles) == 0:
            raise ValueError("Candles data is an empty list.")

        # Check if candles is a list of dictionaries
        if isinstance(candles, list):
            if not all(isinstance(candle, dict) for candle in candles):
                raise ValueError("Candles should be a list of dictionaries.")
            required_keys = {"open", "high", "low", "close"}
            missing_keys = required_keys - set(candles[0].keys())
            if missing_keys:
                raise ValueError(f"Candles are missing required keys: {missing_keys}")
        
        # Check if candles is a pandas DataFrame
        elif isinstance(candles, pd.DataFrame):
            required_columns = ["open", "high", "low", "close"]
            missing_columns = [col for col in required_columns if col not in candles.columns]
            if missing_columns:
                raise ValueError(f"Candles DataFrame is missing required columns: {missing_columns}")
        
        else:
            raise TypeError("Candles should be either a list of dictionaries or a pandas DataFrame.")
        
        # Debug log the structure of the candles
        if isinstance(candles, list):
            logging.info(f"Validated candles (list): {candles[:3]}")
        else:  # DataFrame
            logging.info(f"Validated candles (DataFrame):\n{candles.head()}")

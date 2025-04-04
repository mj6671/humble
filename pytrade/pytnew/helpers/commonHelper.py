import os
import json
import pandas as pd
import logging
from strategies.stop_loss import calculate_stop_loss
from models.model import fetch_pending_orders, fetch_user_api_keys, save_order_to_db, update_users_orders,log_trade,update_order_status, update_trailing_stoploss

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
            filelocation = os.path.join(os.getcwd(), "/minimums.json")
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
            
            if stoploss is not None:
                update_trailing_stoploss(user_id, order_id, stoploss)
                print(f"stoploss updated {stoploss}")
            
        except Exception as e:
            import traceback
            logging.error(f"❌ Error occurred: {e}")
            print(traceback.format_exc())
            return None
            
    def create_trade_conditions(
        candle_condition,
        previous_candle,
        close_condition,
        condition,
        call_candle_condition,
        call_previous_candle,
        call_close_condition,
        call_condition,
        put_candle_condition,
        put_previous_candle,
        put_close_condition,
        put_condition
    ):
        # Define conditions for candles and RSI
        candle_conditions = {
            "Higher Than Previous Candle Close and Above RSI Upper Limit": "RSI > prev_rsi and RSI > UL",
            "Higher Than Previous Candle Close and Above RSI Lower Limit": "RSI > prev_rsi and RSI > LL and RSI < UL",
            "Higher Than Previous Candle Close and Below RSI Lower Limit": "RSI > prev_rsi and RSI < LL",
            "Lower than Previous Candle Close and Above the RSI Upper Limit": "RSI < prev_rsi and RSI > UL",
            "Lower than Previous Candle Close and Below the RSI Upper Limit": "RSI < prev_rsi and RSI > LL and RSI < UL",
            "Lower than Previous Candle Close and Below the RSI Lower Limit": "RSI < prev_rsi and RSI < LL",
        }

        call_candle_conditions = {
            "Higher Than Previous Candle Close and Above RSI Upper Limit": "CRSI > prev_crsi and CRSI > UL",
            "Higher Than Previous Candle Close and Above RSI Lower Limit": "CRSI > prev_crsi and CRSI > LL and CRSI < UL",
            "Higher Than Previous Candle Close and Below RSI Lower Limit": "CRSI > prev_crsi and CRSI < LL",
            "Lower than Previous Candle Close and Above the RSI Upper Limit": "CRSI < prev_crsi and CRSI > UL",
            "Lower than Previous Candle Close and Below the RSI Upper Limit": "CRSI < prev_crsi and CRSI > LL and CRSI < UL",
            "Lower than Previous Candle Close and Below the RSI Lower Limit": "CRSI < prev_crsi and CRSI < LL",
        }

        put_candle_conditions = {
            "Lower than Previous Candle Close and Above the RSI Upper Limit": "prev_prsi > PRSI and PRSI > UL",
            "Lower than Previous Candle Close and Below the RSI Upper Limit": "prev_prsi > PRSI and PRSI < UL and PRSI > LL",
            "Lower than Previous Candle Close and Below the RSI Lower Limit": "prev_prsi > PRSI and PRSI < LL",
            "Higher Than Previous Candle Close and Below RSI Lower Limit": "prev_prsi < PRSI and PRSI < LL",
            "Higher Than Previous Candle Close and Above RSI Lower Limit": "prev_prsi < PRSI and PRSI < UL and PRSI > LL",
            "Higher Than Previous Candle Close and Above RSI Upper Limit": "prev_prsi < PRSI and PRSI > UL",
        }

        # Define the mappings for previous candle, conditions, and close conditions
        previous_candle_dict = {"Open": "O2", "High": "H2", "Low": "L2", "Close": "C2"}
        call_previous_candle_dict = {"Open": "CO2", "High": "CH2", "Low": "CL2", "Close": "CC2"}
        put_previous_candle_dict = {"Open": "PO2", "High": "PH2", "Low": "PL2", "Close": "PC2"}

        condition_dict = {"Open": "O2", "High": "H2", "Low": "L2", "Close": "C2"}
        call_condition_dict = {"Open": "CO2", "High": "CH2", "Low": "CL2", "Close": "CC2"}
        put_condition_dict = {"Open": "PO2", "High": "PH2", "Low": "PL2", "Close": "PC2"}

        close_conditions = {
            "Lower Than or Equal To": "<=",
            "Higher Than or Equal to": ">=",
            "Lower Than": "<",
            "Higher Than": ">",
            "Equal To": "==",
        }

        # Combine conditions
        combined_condition = f"{previous_candle_dict[previous_candle]} {close_conditions[close_condition]} {condition_dict[condition]}"
        call_combined_condition = f"{call_previous_candle_dict[call_previous_candle]} {close_conditions[call_close_condition]} {call_condition_dict[call_condition]}"
        put_combined_condition = f"{put_previous_candle_dict[put_previous_candle]} {close_conditions[put_close_condition]} {put_condition_dict[put_condition]}"

        # Return the generated conditions
        return combined_condition, call_combined_condition, put_combined_condition
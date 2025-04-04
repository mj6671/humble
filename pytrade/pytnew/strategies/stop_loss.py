from decimal import Decimal
import logging
import os
from dotenv import load_dotenv

load_dotenv()

CANDLES_LIMIT = int(os.getenv("CANDLES_LIMIT", 50))  # Default to 50 if not set in the environment

# Function to validate candle data
def validate_candle(candle, keys):
    """
    Validates if the required keys are present in a candle dictionary.
    """
    return all(key in candle for key in keys)

def calculate_variable_percentage_stop_loss(order_price, percentage, order_type):
    """
    Calculates stop-loss based on a variable percentage of the order price.
    """
    percentage = Decimal(percentage)
    if order_type == "BUY":
        return order_price * (Decimal(1) - percentage / Decimal(100))
    elif order_type == "SELL":
        return order_price * (Decimal(1) + percentage / Decimal(100))

def calculate_last_bearish_candle_stop_loss(candles):
    """
    Finds the stop-loss for BUY orders based on the last bearish candle's low.
    """
    required_keys = ["close", "open", "low"]
    for candle in candles[::-1]:
        if validate_candle(candle, required_keys):
            if Decimal(candle["close"]) < Decimal(candle["open"]):  # Bearish candle
                return Decimal(candle["low"])
    logging.warning("No bearish candles found with valid data.")
    return None

def calculate_last_bullish_candle_stop_loss(candles):
    """
    Finds the stop-loss for SELL orders based on the last bullish candle's high.
    """
    required_keys = ["close", "open", "high"]
    for candle in candles[::-1]:
        if validate_candle(candle, required_keys):
            if Decimal(candle["close"]) > Decimal(candle["open"]):  # Bullish candle
                return Decimal(candle["high"])
    logging.warning("No bullish candles found with valid data.")
    return None

def calculate_rsi_based_stop_loss_for_buy(candles, rsi_lower_limit):
    """
    Calculates stop-loss for BUY orders based on the last bearish candle with RSI below the lower limit.
    """
    required_keys = ["rsi", "close", "open", "low"]
    rsi_lower_limit = Decimal(rsi_lower_limit)
    for candle in candles[::-1]:
        if validate_candle(candle, required_keys):
            if Decimal(candle["rsi"]) < rsi_lower_limit and Decimal(candle["close"]) < Decimal(candle["open"]):
                return Decimal(candle["low"])
    logging.warning("No valid RSI-based stop-loss for BUY found.")
    return None

def calculate_rsi_based_stop_loss_for_sell(candles, rsi_upper_limit):
    """
    Calculates stop-loss for SELL orders based on the last bullish candle with RSI above the upper limit.
    """
    required_keys = ["rsi", "close", "open", "high"]
    rsi_upper_limit = Decimal(rsi_upper_limit)
    for candle in candles[::-1]:
        if validate_candle(candle, required_keys):
            if Decimal(candle["rsi"]) > rsi_upper_limit and Decimal(candle["close"]) > Decimal(candle["open"]):
                return Decimal(candle["high"])
    logging.warning("No valid RSI-based stop-loss for SELL found.")
    return None
    
def calculate_lowest_low_for_last_n_candles(candles, n):
    """
    Calculates stop-loss for BUY orders based on the lowest low of the last N candles.
    """
    if len(candles) < n:
        logging.warning(f"Insufficient candles: Expected {n}, got {len(candles)}")
        return None  # Return None if there aren't enough candles

    # Extract the last N candles
    last_n_candles = candles[-n:]

    # Filter candles that have valid 'low' values and ensure they are dictionaries
    valid_lows = []
    for candle in last_n_candles:
        if isinstance(candle, dict) and "low" in candle:
            try:
                valid_lows.append(Decimal(candle["low"]))
            except Exception as e:
                logging.error(f"Error processing candle: {candle}, error: {e}")

    if not valid_lows:
        logging.error("Error calculating lowest low: No valid low values found in the last N candles.")
        return None  # Return None if no valid lows are available

    return min(valid_lows)  # Calculate the lowest low among valid values


def calculate_highest_high_for_last_n_candles(candles, n):
    """
    Calculates stop-loss for SELL orders based on the highest high of the last N candles.
    """
    if len(candles) < n:
        logging.warning(f"Insufficient candles: Expected {n}, got {len(candles)}")
        return None  # Return None if there aren't enough candles

    # Extract the last N candles
    last_n_candles = candles[-n:]

    # Filter candles that have valid 'high' values and ensure they are dictionaries
    valid_highs = []
    for candle in last_n_candles:
        if isinstance(candle, dict) and "high" in candle:
            try:
                valid_highs.append(Decimal(candle["high"]))
            except Exception as e:
                logging.error(f"Error processing candle: {candle}, error: {e}")

    if not valid_highs:
        logging.error("Error calculating highest high: No valid high values found in the last N candles.")
        return None  # Return None if no valid highs are available

    return max(valid_highs)  # Calculate the highest high among valid values   

def calculate_stop_loss(order_price, percentage, candles, rsi_lower_limit, rsi_upper_limit, order_type):
    """
    Combines all the stop-loss calculations and applies combination logic dynamically.
    """
    order_price = Decimal(order_price)
    percentage_based_stop_loss = calculate_variable_percentage_stop_loss(order_price, percentage, order_type)

    if order_type == "BUY":
        last_candle_stop_loss = calculate_last_bearish_candle_stop_loss(candles)
        rsi_stop_loss = calculate_rsi_based_stop_loss_for_buy(candles, rsi_lower_limit)
        lowest_low_last_3_candles = calculate_lowest_low_for_last_n_candles(candles, 3)
        lowest_low_last_2_candles = calculate_lowest_low_for_last_n_candles(candles, 2)
        stop_loss_candidates = [last_candle_stop_loss, rsi_stop_loss, lowest_low_last_3_candles, lowest_low_last_2_candles]
    elif order_type == "SELL":
        last_candle_stop_loss = calculate_last_bullish_candle_stop_loss(candles)
        rsi_stop_loss = calculate_rsi_based_stop_loss_for_sell(candles, rsi_upper_limit)
        highest_high_last_3_candles = calculate_highest_high_for_last_n_candles(candles, 3)
        highest_high_last_2_candles = calculate_highest_high_for_last_n_candles(candles, 2)
        stop_loss_candidates = [last_candle_stop_loss, rsi_stop_loss, highest_high_last_3_candles, highest_high_last_2_candles]
    else:
        logging.error(f"Invalid order type: {order_type}")
        return None

    # Filter valid stop-loss candidates
    stop_loss_candidates = [sl for sl in stop_loss_candidates if sl is not None]

    if not stop_loss_candidates:
        logging.error("No valid stop-loss candidates available")
        return None

    # Apply combination logic
    if order_type == "BUY":
        return max(min(stop_loss_candidates), percentage_based_stop_loss)
    elif order_type == "SELL":
        return min(max(stop_loss_candidates), percentage_based_stop_loss)

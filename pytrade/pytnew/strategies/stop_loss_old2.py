from decimal import Decimal
import logging
import os
from dotenv import load_dotenv

load_dotenv()


CANDLES_LIMIT= int(os.getenv("CANDLES_LIMIT"))

def calculate_variable_percentage_stop_loss(order_price, percentage, order_type):
    
    percentage = Decimal(percentage)  # Convert percentage to Decimal
    if order_type == "BUY":
        return order_price * (Decimal(1) - percentage / Decimal(100))  # Stop-loss is below for BUY orders
    elif order_type == "SELL":
        return order_price * (Decimal(1) + percentage / Decimal(100))  # Stop-loss is above for SELL orders


def calculate_last_bearish_candle_stop_loss(candles):
   
   
   
    for candle in reversed(candles):
        if Decimal(candle["close"]) < Decimal(candle["open"]):  # Bearish candle
            return Decimal(candle["low"])

def calculate_last_bullish_candle_stop_loss(candles):
    """
    Calculates stop-loss for SELL orders based on the last bullish candle's high.
    """
    for candle in reversed(candles):
        if Decimal(candle["close"]) > Decimal(candle["open"]):  # Bullish candle
            return Decimal(candle["high"])

def calculate_rsi_based_stop_loss_for_buy(candles, rsi_lower_limit):
    """
    Calculates stop-loss for BUY orders based on the last bearish candle with RSI below the lower limit.
    """
    rsi_lower_limit = Decimal(rsi_lower_limit)
    for i in reversed(range(len(candles))):
        if Decimal(candles[i]["rsi"]) < rsi_lower_limit and Decimal(candles[i]["close"]) < Decimal(candles[i]["open"]):
            return Decimal(candles[i]["low"])

def calculate_rsi_based_stop_loss_for_sell(candles, rsi_upper_limit):
    """
    Calculates stop-loss for SELL orders based on the last bullish candle with RSI above the upper limit.
    """
    rsi_upper_limit = Decimal(rsi_upper_limit)
    for i in reversed(range(len(candles))):
        if Decimal(candles[i]["rsi"]) > rsi_upper_limit and Decimal(candles[i]["close"]) > Decimal(candles[i]["open"]):
            return Decimal(candles[i]["high"])

def calculate_lowest_low_for_last_n_candles(candles, n):
    """
    Calculates stop-loss for BUY orders based on the lowest low of the last N candles.
    """
    if len(candles) < n:
        logging.warning(f"Insufficient candles: Expected {n}, got {len(candles)}")
        return None  # Or handle gracefully
    
    
    last_n_candles = candles[-n:]
    return min(Decimal(candle["low"]) for candle in last_n_candles)

def calculate_highest_high_for_last_n_candles(candles, n):
    """
    Calculates stop-loss for SELL orders based on the highest high of the last N candles.
    """
    if len(candles) < n:
        logging.warning(f"Insufficient candles: Expected {n}, got {len(candles)}")
        return None  # Or handle gracefully
    
    
    last_n_candles = candles[-n:]
    return max(Decimal(candle["high"]) for candle in last_n_candles)

def calculate_stop_loss(order_price, percentage, candles, rsi_lower_limit, rsi_upper_limit, order_type):
    """
    Combines all the stop-loss calculations and applies combination logic dynamically.
    """
    order_price = Decimal(order_price)
    percentage = Decimal(percentage)
    percentage_based_stop_loss = calculate_variable_percentage_stop_loss(order_price, percentage, order_type)

    if order_type == "BUY":
        last_candle_stop_loss = calculate_last_bearish_candle_stop_loss(candles)
        rsi_stop_loss = calculate_rsi_based_stop_loss_for_buy(candles, rsi_lower_limit)
        lowest_low_last_3_candles = calculate_lowest_low_for_last_n_candles(candles, 3)
        lowest_low_last_2_candles = calculate_lowest_low_for_last_n_candles(candles, 2)
        stop_loss_candidates = [
            last_candle_stop_loss, rsi_stop_loss, lowest_low_last_3_candles, lowest_low_last_2_candles
        ]

        # Remove None values and find the lowest stop-loss
        stop_loss_candidates = [sl for sl in stop_loss_candidates if sl is not None]
        
        if not stop_loss_candidates:
            logging.error("No valid stop-loss candidates available")
            return None  # Or set a default value
        
        lowest_stop_loss = min(stop_loss_candidates)

        # Use combination logic
        return max(lowest_stop_loss, percentage_based_stop_loss)

    elif order_type == "SELL":
        last_candle_stop_loss = calculate_last_bullish_candle_stop_loss(candles)
        rsi_stop_loss = calculate_rsi_based_stop_loss_for_sell(candles, rsi_upper_limit)
        highest_high_last_3_candles = calculate_highest_high_for_last_n_candles(candles, 3)
        highest_high_last_2_candles = calculate_highest_high_for_last_n_candles(candles, 2)
        stop_loss_candidates = [
            last_candle_stop_loss, rsi_stop_loss, highest_high_last_3_candles, highest_high_last_2_candles
        ]

        # Remove None values and find the highest stop-loss
        stop_loss_candidates = [sl for sl in stop_loss_candidates if sl is not None]
        
        if not stop_loss_candidates:
            logging.error("No valid stop-loss candidates available")
            return None  # Or set a default value
        
        
        highest_stop_loss = max(stop_loss_candidates)

        # Use combination logic
        return min(highest_stop_loss, percentage_based_stop_loss)

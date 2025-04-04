import numpy as np
import pandas as pd
import pandas_ta as ta
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

SUPERTREND = os.getenv("SUPERTREND")

# Calculate RSI
def calculate_rsi(df, period):
    df["rsi"] = ta.rsi(df["close"], length=period)
    df["rsi"] = df["rsi"].fillna(0)  # Replace NaN with 0
    return df

# Calculate Supertrend
def calculate_supertrend(df, period, multiplier):
    st = df.ta.supertrend(length=period, multiplier=multiplier)
    df["supertrend"] = st.iloc[:, 0]
    df["supertrend"] = df["supertrend"].fillna(0)  # Replace NaN with 0
    df["trend"] = np.where(df["close"] > df["supertrend"], "green", "red")
    return df

# Check entry conditions
def check_entry_conditions(df):
    last_candle = df.iloc[-2]
    prev_candle = df.iloc[-3]
    
    # Candle Condition
    is_green = last_candle["close"] > last_candle["open"]
    is_red = last_candle["open"] > last_candle["close"]
    candle_color = "green" if is_green else "red"

    # RSI Condition
    rsi_up = last_candle["rsi"] > prev_candle["rsi"]
    rsi_down = last_candle["rsi"] < prev_candle["rsi"]
    rsi_movement = "up" if rsi_up else "down"

    # Supertrend Condition
    supertrend_green = last_candle["trend"] == "green"
    supertrend_red = last_candle["trend"] == "red"

    # Determine Action
    if SUPERTREND == "yes":
        supertrend_status = "yes"
        supertrend_trend = "green" if supertrend_green else "red"
        if is_green and rsi_up and supertrend_green:
            action = "BUY"
        elif is_red and rsi_down and supertrend_red:
            action = "SELL"
        else:
            action = "HOLD"
    else:
        supertrend_status = "no"
        supertrend_trend = ""  # Empty when Supertrend is disabled
        if is_green and rsi_up:
            action = "BUY"
        elif is_red and rsi_down:
            action = "SELL"
        else:
            action = "HOLD"

    # JSON Data for Logging
    conditions_json = json.dumps({
        "candlestick": {
            "color": candle_color,
            "open": last_candle["open"],
            "close": last_candle["close"],
            "high": last_candle["high"],
            "low": last_candle["low"],
            "volume": last_candle["volume"]
        },
        "rsi": {
            "movement": rsi_movement,
            "last_candle": last_candle["rsi"],
            "prev_candle": prev_candle["rsi"]
        },
        "supertrend": {
            "enabled": supertrend_status,
            "trend": supertrend_trend  # Empty if Supertrend is disabled
        }
    })

    return action, conditions_json
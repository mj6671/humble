import itertools
import json

# Define possible values
previous_candle_dict = {"Open", "High", "Low", "Close"}
condition_dict = {"Open", "High", "Low", "Close"}
close_conditions = {"Lower Than or Equal To", "Higher Than or Equal to", "Lower Than", "Higher Than", "Equal To"}

# Define initial RSI conditions
initial_candle_conditions = {
    "Higher Than Previous Candle Close and Above RSI Upper Limit": "RSI > prev_rsi and RSI > UL",
    "Higher Than Previous Candle Close and Above RSI Lower Limit": "RSI > prev_rsi and RSI > LL and RSI < UL",
    "Higher Than Previous Candle Close and Below RSI Lower Limit": "RSI > prev_rsi and RSI < LL",
    "Lower than Previous Candle Close and Above the RSI Upper Limit": "RSI < prev_rsi and RSI > UL",
    "Lower than Previous Candle Close and Below the RSI Upper Limit": "RSI < prev_rsi and RSI > LL and RSI < UL",
    "Lower than Previous Candle Close and Below the RSI Lower Limit": "RSI < prev_rsi and RSI < LL",
}

# Generate trade conditions
trade_conditions = []
index = 1

for previous_candle, close_condition, condition in itertools.product(previous_candle_dict, close_conditions, condition_dict):
    for rsi_text, rsi_condition in initial_candle_conditions.items():
        
        # Construct trade condition text
        trade_condition_text = f"Current Candle {condition} is {close_condition} Previous Candle {previous_candle} and {rsi_text.split(' and ')[-1]}"

        trade_conditions.append({
            "id": index,
            "previous_candle": previous_candle,
            "close_condition": close_condition,
            "condition": condition,
            "trade_condition": trade_condition_text,
            "rsi_condition": rsi_condition
        })
        index += 1

# Save to JSON file
with open("trade_conditions.json", "w") as json_file:
    json.dump(trade_conditions, json_file, indent=4)
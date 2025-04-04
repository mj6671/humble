from binance.spot import Spot
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

APIKEY ='s5XXbDfBkDLoDrvLQyH3XqeOpqvWjVBC5KM1wOJmKYpVVht3xewzSY0mn8Tr1YMk';
APISECRET ='WR5nRuoFpLZMOxZbJeMy00awWDvCXp4cT68oeJCShLkj9uSd0Qd5rRoKwIwGbDws';

# Initialize Binance client
client = Spot(api_key=APIKEY, api_secret=APISECRET)

try:
    # Fetch exchange info
    exchange_info = client.exchange_info()
    minimums = {}

    for symbol_info in exchange_info['symbols']:
        filters = {
            "minNotional": "0.00010000",
            "maxNotional": "9000000.00000000",
            "minQty": "0.00100000",
            "maxQty": "100000.00000000",
            "stepSize": "0.00100000",
            "minPrice": "0.00000100",
            "maxPrice": "100000.00000000",
            "minTrailingAboveDelta": 10,
            "maxTrailingAboveDelta": 2000, 
            "minTrailingBelowDelta": 10, 
            "maxTrailingBelowDelta": 100000
        }

        for filter_obj in symbol_info['filters']:
            if filter_obj['filterType'] == "NOTIONAL":
                filters['minNotional'] = "{:.8f}".format(float(filter_obj['minNotional']))
                filters['maxNotional'] = "{:.8f}".format(float(filter_obj['maxNotional']))
            elif filter_obj['filterType'] == "PRICE_FILTER":
                filters['minPrice'] = "{:.8f}".format(float(filter_obj['minPrice']))
                filters['maxPrice'] = "{:.8f}".format(float(filter_obj['maxPrice']))
            elif filter_obj['filterType'] == "LOT_SIZE":
                filters['minQty'] = "{:.8f}".format(float(filter_obj['minQty']))
                filters['maxQty'] = "{:.8f}".format(float(filter_obj['maxQty']))
                filters['stepSize'] = "{:.8f}".format(float(filter_obj['stepSize']))
            elif filter_obj['filterType'] == "TRAILING_DELTA":
                filters['minTrailingAboveDelta'] = str(filter_obj['minTrailingAboveDelta'])
                filters['maxTrailingAboveDelta'] = str(filter_obj['maxTrailingAboveDelta'])
                filters['minTrailingBelowDelta'] = str(filter_obj['minTrailingBelowDelta'])
                filters['maxTrailingBelowDelta'] = str(filter_obj['maxTrailingBelowDelta'])

        minimums[symbol_info['symbol']] = filters

    # Save to file
    with open("minimums.json", "w") as file:
        json.dump(minimums, file, indent=4)
    print("Minimums saved to minimums.json")

except Exception as e:
    print(f"Error fetching exchange info: {e}")

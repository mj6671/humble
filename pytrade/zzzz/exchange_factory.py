from services.exchanges.binance_api import BinanceAPI
#from services.exchanges.kucoin_api import KuCoinAPI  # Assuming you'll add KuCoinAPI

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT")
BINANCE_TESTNET_API = os.getenv("BINANCE_TESTNET_API")
BINANCE_TESTNET_SECRET = os.getenv("BINANCE_TESTNET_SECRET")

def get_exchange_client(exchange, api_key, api_secret):
    """Returns the correct exchange API client"""
    if exchange.lower() == "binance":
        if ENVIRONMENT == "production":
            print(f"production")
            apibaseurl = "https://api.binance.com"
            return BinanceAPI(api_key, api_secret, base_url=apibaseurl)
        elif ENVIRONMENT == "development":
            print(f"development")
            apibaseurl = "https://testnet.binance.vision"
            return BinanceAPI(BINANCE_TESTNET_API, BINANCE_TESTNET_SECRET, base_url=apibaseurl)
        else:  # Default to production if ENV is not recognized
            apibaseurl = "https://api.binance.com"
            return BinanceAPI(api_key, api_secret, base_url=apibaseurl)
    else:
       raise ValueError(f"Unsupported exchange: {exchange}")    
        
    #elif exchange.lower() == "kucoin":
        #return KuCoinAPI(api_key, api_secret)  # Support for KuCoin
    #else:
       # raise ValueError(f"Unsupported exchange: {exchange}")

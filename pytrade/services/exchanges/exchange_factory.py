from services.exchanges.binance_api import BinanceAPI
#from services.exchanges.kucoin_api import KuCoinAPI  # Assuming KuCoinAPI implementation
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT")
BINANCE_TESTNET_API = os.getenv("BINANCE_TESTNET_API")
BINANCE_TESTNET_SECRET = os.getenv("BINANCE_TESTNET_SECRET")


class ExchangeFactory:
    @staticmethod
    def get_exchange_client(exchange_id, api_key, api_secret):
        if exchange_id == "1":
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
        #elif exchange_id == "2":
            #return KuCoinAPI(api_key, api_secret)  # KuCoin API implementation
        else:
            raise ValueError(f"Unsupported exchange ID: {exchange_id}")


import pandas as pd
from binance.spot import Spot
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
import logging
from dotenv import load_dotenv
from decimal import Decimal, ROUND_DOWN
import os

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

ENVIRONMENT = os.getenv("ENVIRONMENT")

class BinanceAPI:
    def __init__(self, api_key, api_secret,base_url):
        """Initialize Binance client with API keys and base URL."""
        self.client = Spot(api_key, api_secret, base_url=base_url)  # ✅ Use base_url

        # WebSockets do NOT need base_url
        self.ws_api_client = SpotWebsocketAPIClient(on_message=self.api_message_handler)
        self.ws_stream_client = SpotWebsocketStreamClient(on_message=self.stream_message_handler)
    
    def api_message_handler(self, _, message):
        """Handle WebSocket API messages."""
        logging.info(f"WebSocket API Message: {message}")

    def stream_message_handler(self, _, message):
        """Handle WebSocket Stream messages."""
        logging.info(f"WebSocket Stream Message: {message}")
    
    def create_market_order(self, side, symbol, quantity):
        """Execute a market order on Binance."""
        try:
            order = self.client.new_order(
                symbol=symbol,
                side=side.upper(),
                type="MARKET",
                quantity=quantity
            )
            return order
        except Exception as e:
            logging.error(f"Error executing market order: {e}")
            return None
    
    def retrieve_candlestick_data(self, symbol, interval, limit=100):
        """Retrieve historical candlestick data from Binance."""
        try:
            klines = self.client.klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                "time", "open", "high", "low", "close", "volume", "close_time", 
                "qav", "num_trades", "taker_base", "taker_quote", "ignore"
            ])
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
            return df
        except Exception as e:
            logging.error(f"Error retrieving candlestick data: {e}")
            return None
    
    def fetch_account_details(self):
        """Retrieve account details."""
        try:
            return self.client.account()
        except Exception as e:
            logging.error(f"Error retrieving account details: {e}")
            return None
    
    def check_balance(self, asset):
        """Check the balance of a specific asset."""
        try:
            balances = self.client.account()["balances"]
            for balance in balances:
                if balance["asset"] == asset:
                    return balance
            return None
        except Exception as e:
            logging.error(f"Error checking balance: {e}")
            return None
    
    def get_order_book_data(self, symbol, limit=100):
        """Retrieve order book data."""
        try:
            return self.client.depth(symbol=symbol, limit=limit)
        except Exception as e:
            logging.error(f"Error retrieving order book data: {e}")
            return None
    
    def fetch_recent_trades(self, symbol, limit=500):
        """Retrieve recent trades for a symbol."""
        try:
            return self.client.trades(symbol=symbol, limit=limit)
        except Exception as e:
            logging.error(f"Error retrieving recent trades: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Retrieve the latest price of a symbol."""
        try:
            return self.client.ticker_price(symbol=symbol)
        except Exception as e:
            logging.error(f"Error retrieving current price: {e}")
            return None
    
    def fetch_24h_statistics(self, symbol):
        """Retrieve 24-hour price change statistics."""
        try:
            return self.client.ticker_24hr(symbol=symbol)
        except Exception as e:
            logging.error(f"Error retrieving 24h statistics: {e}")
            return None
    
    def list_open_orders(self, symbol):
        """Retrieve open orders for a symbol."""
        try:
            return self.client.get_open_orders(symbol=symbol)
        except Exception as e:
            logging.error(f"Error retrieving open orders: {e}")
            return None
    
    def remove_order(self, symbol, order_id):
        """Cancel an order using its order ID."""
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logging.error(f"Error canceling order: {e}")
            return None
    
    def stream_ticker_updates(self, symbol):
        """Stream live ticker price updates using WebSocket Stream."""
        self.ws_stream_client.ticker(symbol=symbol, type="FULL")
    
    def stop_websockets(self):
        """Stop both WebSocket clients."""
        self.ws_api_client.stop()
        self.ws_stream_client.stop()

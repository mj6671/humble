import ccxt
import datetime

class BinanceAPI:
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'spot'}
        })
    
    def create_market_order(self, symbol, side, amount):
        """Place a market order."""
        return self.exchange.create_order(symbol, 'market', side, amount)
    
    def create_limit_order(self, symbol, side, amount, price):
        """Place a limit order."""
        return self.exchange.create_order(symbol, 'limit', side, amount, price)
    
    def get_order_details(self, order_id, symbol):
        """Fetch order details by order ID."""
        return self.exchange.fetch_order(order_id, symbol)
    
    def fetch_latest_price(self, symbol):
        """Get the latest market price for a symbol."""
        ticker = self.exchange.fetch_ticker(symbol)
        return ticker['last']
    
    def fetch_historical_data(self, symbol, timeframe='1h', since=None, limit=100):
        """Fetch historical OHLCV data."""
        since = since or self.exchange.parse8601(datetime.datetime.utcnow().isoformat())
        return self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    
    def modify_order(self, order_id, symbol, price=None, amount=None):
        """Modify an open order."""
        return self.exchange.edit_order(order_id, symbol, 'limit', price=price, amount=amount)
    
    def cancel_order(self, order_id, symbol):
        """Cancel an order by order ID."""
        return self.exchange.cancel_order(order_id, symbol)
    
    def fetch_open_orders(self, symbol=None):
        """Fetch all open orders for a given symbol (or all symbols)."""
        return self.exchange.fetch_open_orders(symbol)
    
    def fetch_balance(self):
        """Fetch account balance."""
        return self.exchange.fetch_balance()

# Example Usage
# api_key = "your_api_key"
# api_secret = "your_api_secret"
# binance_helper = BinanceAPI(api_key, api_secret)
# print(binance_helper.fetch_latest_price('BTC/USDT'))

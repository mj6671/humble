import pandas as pd
from binance.spot import Spot
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

ENVIRONMENT = os.getenv("ENVIRONMENT")

class BinanceAPI:
    def __init__(self, api_key, api_secret, base_url):
        """Initialize Binance client with API keys and base URL."""
        self.client = Spot(api_key, api_secret, base_url=base_url)
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
            
    def create_new_order(self, side, symbol, type, quantity, price):
        """Execute a limit order on Binance."""
        try:
            order = self.client.new_order(
                symbol=symbol,
                side=side.upper(),
                type=type.upper(),
                price=price,
                quantity=quantity,
                timeInForce='GTC'
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

    def cancel_open_orders(self, symbol):
        """Cancel all open orders for a symbol."""
        try:
            return self.client.cancel_open_orders(symbol=symbol)
        except Exception as e:
            logging.error(f"Error canceling order: {e}")
            return None

    def remove_order(self, symbol, order_id):
        """Cancel an order using its order ID."""
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            logging.error(f"Error canceling order: {e}")
            return None
    
    def get_order(self, symbol, orderId):
        """Retrieve details of a specific order."""
        try:
            order = self.client.get_order(
                symbol=symbol,
                orderId=orderId
            )
            return order
        except Exception as e:
            logging.error(f"Error retrieving order: {e}")
            return None

    def stream_ticker_updates(self, symbol):
        """Stream live ticker price updates using WebSocket Stream."""
        self.ws_stream_client.ticker(symbol=symbol, type="FULL")
    
    def stop_websockets(self):
        """Stop both WebSocket clients."""
        self.ws_api_client.stop()
        self.ws_stream_client.stop()
        
    def modify_stop_loss(self, symbol, order_id, new_stop_price, price, quantity, all_exited):
        """Modify an existing stop-loss order."""
        try:
            # Step 1: Fetch existing order details (Optional)
            existing_order = self.client.get_order(symbol=symbol, orderId=order_id)
            print(f"Existing Order: {existing_order}")
            side = existing_order["side"]
            
            # Step 2: Cancel existing stop-loss
            cancel_response = self.client.cancel_order(symbol=symbol, orderId=order_id)
            print(f"Cancelled Order: {cancel_response}")

            # Step 3: Place new stop-loss
            new_order = self.client.new_order(
                symbol=symbol,
                side=side,
                type="STOP_LOSS_LIMIT",
                timeInForce="GTC",
                quantity=quantity,
                price=price,
                stopPrice=new_stop_price
            )
            return new_order["orderId"], all_exited
        except Exception as e:
            logging.error(f"Error modifying stop-loss order: {e}")
            return None

    def create_slm_order(self, side, symbol, quantity, stopPrice, limit_price=None):
        """Create a stop-loss market/limit order."""
        try:
            order_params = {
                'symbol': symbol,
                'side': side.upper(),
                'quantity': quantity,
                'stopPrice': stopPrice
            }
            
            if limit_price is None:
                order_params['type'] = 'STOP_LOSS'
            else:
                if limit_price <= 0:
                    raise ValueError("Limit price must be positive")
                order_params['type'] = 'STOP_LOSS_LIMIT'
                order_params['price'] = limit_price
                order_params['timeInForce'] = 'GTC'  # Added for STOP_LOSS_LIMIT

            order = self.client.new_order(**order_params)
            return order
        except Exception as e:
            logging.error(f"Error executing stop-loss order: {e}")
            return None

    def modify_mkt_order(self, sl_order_id, local_quantity, symbol):
        """Modify an existing stop-loss order with a new quantity."""
        try:
            # Fetch the existing order details
            order = self.client.get_order(symbol=symbol, orderId=sl_order_id)
            
            if order['status'] in ['NEW', 'PARTIALLY_FILLED']:
                # Cancel the existing stop-loss order
                self.client.cancel_order(symbol=symbol, orderId=sl_order_id)
                
                # Place a new stop-loss order with updated quantity
                response = self.client.new_order(
                    symbol=symbol,
                    side=order['side'],
                    type=order['type'],
                    quantity=local_quantity,
                    stopPrice=order.get('stopPrice'),
                    price=order.get('price'),
                    timeInForce='GTC'
                )
                print("Order modified successfully:", response)
                return response
            else:
                print("Order cannot be modified as it is already executed or canceled.")
                return None
        except Exception as e:
            print("Error modifying order:", str(e))
            return None

    def get_order_status(self, symbol, orderId):
        """Retrieve the status of a specific order."""
        try:
            order = self.client.get_order(symbol=symbol, orderId=orderId)
            status = order.get("status")
            return status
        except Exception as e:
            logging.error(f"Error retrieving order: {e}")
            return None

    def modify_sl_order_for_sell(self, symbol, sl_oid, single_candle_sl, qty, all_exited):
        """Modify stop-loss order for a sell position."""
        try:
            # Cancel existing stop-loss order
            cancel_response = self.client.cancel_order(symbol=symbol, orderId=sl_oid)
            print(f"Cancelled Order: {cancel_response}")
            ticker = self.client.ticker_price(symbol=symbol)
            current_price = float(ticker['price'])
            print(f"Current market price for {symbol}: {current_price}")

            # Ensure stopPrice is below current price for a sell stop-loss
            stop_price = float(single_candle_sl)
            if stop_price >= current_price:
                raise ValueError(f"Stop price {stop_price} must be below current price {current_price} for a sell stop-loss.")

            # Set limit price slightly below stop price (e.g., 0.1% lower)
            limit_price = stop_price * 0.999
            # Create a new stop-loss order with updated price
            order = self.client.new_order(
                symbol=symbol,
                side="SELL",
                type="STOP_LOSS_LIMIT",
                quantity=qty,
                price=str(round(limit_price, 2)),  # Stop-loss limit price
                stopPrice=str(round(stop_price, 2)),   # Trigger price
                timeInForce="GTC"
            )

            print(f"Stop-loss order modified successfully. New Order ID: {order['orderId']}")
            
            return order["orderId"], all_exited

        except Exception as e:
            print(f"Error modifying stop-loss order {sl_oid}: {str(e)}")
            return None, all_exited
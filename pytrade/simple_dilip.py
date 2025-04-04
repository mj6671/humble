import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import datetime
import bisect
from dotenv import load_dotenv
import os
from models.model import fetch_pending_orders, fetch_user_api_keys, save_order_to_db, update_users_orders,log_trade,update_order_status,get_open_order
#from datetime import datetime
from config.database import db_connect
from decimal import Decimal, ROUND_DOWN
import math
from helpers.commonHelper import commonHelper
from services.exchanges.exchange_factory import ExchangeFactory
load_dotenv()

global Isl_percentage
global RSI_LOWER
global RSI_UPPER
global RSI_PERIOD

EXIT_TIME = os.getenv("EXIT_TIME")
RSI_PERIOD = int(os.getenv("RSI_PERIOD"))
RSI_LENGTH = int(os.getenv("RSI_LENGTH"))
RSI_UPPER = int(os.getenv("RSI_UPPER"))
RSI_MIDDLE = int(os.getenv("RSI_MIDDLE"))
RSI_LOWER = int(os.getenv("RSI_LOWER"))
PUT_UL = int(os.getenv("PUT_UL"))
PUT_LL = int(os.getenv("PUT_LL"))
TIMEFRAME = os.getenv("TIMEFRAME")
CHART_REFERENCE = os.getenv("CHART_REFERENCE")
THIRD_CHART_REFERENCE = os.getenv("THIRD_CHART_REFERENCE")

candle_condition=os.getenv("candle_condition")
previous_candle=os.getenv("previous_candle")
close_condition=os.getenv("close_condition")
condition=os.getenv("condition")
call_candle_condition=os.getenv("call_candle_condition")
call_previous_candle=os.getenv("call_previous_candle")
call_close_condition=os.getenv("call_close_condition")
call_condition=os.getenv("call_condition")
put_candle_condition=os.getenv("put_candle_condition")
put_previous_candle=os.getenv("put_previous_candle")
put_close_condition=os.getenv("put_close_condition")
put_condition=os.getenv("put_condition")

SUPERTREND=os.getenv("SUPERTREND")

ORDER_TYPE=os.getenv("ORDER_TYPE") #market slm limit
BUY_ONLY_CALL_OR_PUT=os.getenv("BUY_ONLY_CALL_OR_PUT")

Isl_percentage = os.getenv("Isl_percentage", "10%")  

length = os.getenv("length")
factor = os.getenv("factor")


BULLISH_RSI_ENABLER = os.getenv("BULLISH_RSI_ENABLER")
BULLISH_RSI_LIMIT = os.getenv("BULLISH_RSI_LIMIT")
BULLISH_CONDITION = os.getenv("BULLISH_CONDITION")

TARGET_SPLIT = os.getenv("TARGET_SPLIT")
TARGET = os.getenv("TARGET")
TARGET = float(TARGET.strip('%')) / 100
TSL1 = os.getenv("TSL1")
TSL2 = os.getenv("TSL2")

FIRST_TARGET = os.getenv("FIRST_TARGET")
FIRST_TARGET = float(FIRST_TARGET.strip('%')) / 100
FIRST_TARGET_TRAILING = os.getenv("FIRST_TARGET_TRAILING")
FIRST_TARGET_TRAILING = float(FIRST_TARGET_TRAILING.strip('%')) / 100
SECOND_TARGET = os.getenv("SECOND_TARGET")
SECOND_TARGET = float(SECOND_TARGET.strip('%')) / 100
SECOND_TARGET_TRAILING = os.getenv("SECOND_TARGET_TRAILING")
SECOND_TARGET_TRAILING = float(SECOND_TARGET_TRAILING.strip('%')) / 100

SINGLE_CANDLE_CONDITION = os.getenv("SINGLE_CANDLE_CONDITION")
TSLOFSCC = os.getenv("TSLOFSCC")
TSLOFSCC = float(TSLOFSCC.strip('%')) / 100
AFTER_SCC_X_PCT_PRICE_MOVE = os.getenv("AFTER_SCC_X_PCT_PRICE_MOVE")
AFTER_SCC_X_PCT_PRICE_MOVE = float(AFTER_SCC_X_PCT_PRICE_MOVE.strip('%')) / 100
AFTER_SCC_Y_PCT_TRAILING_MOVE = os.getenv("AFTER_SCC_Y_PCT_TRAILING_MOVE")
AFTER_SCC_Y_PCT_TRAILING_MOVE = float(AFTER_SCC_Y_PCT_TRAILING_MOVE.strip('%')) / 100

BINANCE_TESTNET_API = os.getenv("BINANCE_TESTNET_API")
BINANCE_TESTNET_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
# Binance API setup
binance = ccxt.binance({
    'apiKey': BINANCE_TESTNET_API,
    'secret': BINANCE_TESTNET_SECRET,
    'options': {'defaultType': 'spot'},
})
# Connect to Testnet Endpoint
binance.set_sandbox_mode(True)  # Enables testnet mode


binance2 = ExchangeFactory.get_exchange_client("1", BINANCE_TESTNET_API, BINANCE_TESTNET_SECRET)
# Function to fetch OHLCV data
def fetch_ohlcv(symbol):
    #bars = binance.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
    bars = binance2.retrieve_candlestick_data(symbol, TIMEFRAME, limit=50)
    df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    
    return df

# Function to convert to Heikin-Ashi candles
def convert_to_heikin_ashi(df):
    ha_df = df.copy()
    
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha_df['high'] = df[['high', 'open', 'close']].max(axis=1)
    ha_df['low'] = df[['low', 'open', 'close']].min(axis=1)

    ha_df.dropna(inplace=True)  # Remove first row due to NaN from shift
    return ha_df

# Function to calculate RSI
def calculate_rsi(data, period=RSI_PERIOD):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_rsi_df(new_data_df,rsi_length):
	RSI,prev_rsi,rsi_df = calculate_last_rsi(new_data_df,rsi_length)
	return RSI,prev_rsi,rsi_df
def calculate_last_rsi(ndf, time_period):
        df= heikin_ashi_df(ndf)
        
      #  df['rsi']  = df.ta.rsi(df['close'], timeperiod=time_period)
        df["rsi"] = df.ta.rsi(close=df["close"], length=time_period)  # 'length' is the period
       
       
        last_row_rsi = df['rsi'].iloc[-1]
        previous_row_rsi= df['rsi'].iloc[-2]
    
        return last_row_rsi,previous_row_rsi,df
		

def heikin_ashi_df(rel_df, last_row=False):
    rel_df = rel_df.rename(columns={'timestamp': 'date'})

    if 'date' not in rel_df.columns:
        raise ValueError(f"Missing 'date' column in DataFrame. Available columns: {list(rel_df.columns)}")

    if rel_df.empty:
        return rel_df  # Return empty DataFrame if input is empty

    if not last_row:
        rel_df = rel_df.iloc[:-1]  # Drop the last row safely

    date_list = rel_df['date'].tolist()  # Extract dates
    pd.options.mode.chained_assignment = None  # Suppress warnings

    # Create a copy to avoid modifying the original DataFrame
    HAdf = rel_df[['open', 'high', 'low', 'close']].copy()

    # Compute Heikin-Ashi close prices
    HAdf['close'] = round((rel_df[['open', 'high', 'low', 'close']].sum(axis=1)) / 4, 2)

    # Compute Heikin-Ashi open prices using vectorized operations
    HAdf['open'] = (HAdf['close'].shift(1) + HAdf['open'].shift(1)) / 2
    HAdf.iloc[0, 0] = round((rel_df.iloc[0]['open'] + rel_df.iloc[0]['close']) / 2, 2)  # First row exception

    # Compute Heikin-Ashi high and low
    HAdf['high'] = HAdf[['open', 'close']].join(rel_df[['high']]).max(axis=1)
    HAdf['low'] = HAdf[['open', 'close']].join(rel_df[['low']]).min(axis=1)

    # Assign date column and reorder columns
    HAdf.insert(0, 'date', date_list)

    return HAdf


		
def heikin_ashi_df_new(rel_df, last_row=False):
    ha_df = rel_df.copy()
    
    # Initialize Heikin-Ashi columns
    ha_df["HA_Close"] = (ha_df["open"] + ha_df["high"] + ha_df["low"] + ha_df["close"]) / 4
    ha_df["HA_Open"] = 0.0
    ha_df["HA_High"] = 0.0
    ha_df["HA_Low"] = 0.0

    # Set the first HA_Open value (assume first Open is same as real Open)
    ha_df.loc[0, "HA_Open"] = (ha_df.loc[0, "open"] + ha_df.loc[0, "close"]) / 2

    # Compute Heikin-Ashi candles
    for i in range(1, len(ha_df)):
        ha_df.loc[i, "HA_Open"] = (ha_df.loc[i - 1, "HA_Open"] + ha_df.loc[i - 1, "HA_Close"]) / 2
        ha_df.loc[i, "HA_High"] = max(ha_df.loc[i, "high"], ha_df.loc[i, "HA_Open"], ha_df.loc[i, "HA_Close"])
        ha_df.loc[i, "HA_Low"] = min(ha_df.loc[i, "low"], ha_df.loc[i, "HA_Open"], ha_df.loc[i, "HA_Close"])

    # Return last row if needed
    if last_row:
        return ha_df.iloc[-1]

    return ha_df

def qty_to_exit(symbol, local_quantity):
    try:
        # Fetch symbol filters (lot size/step size)
        get_minimums = commonHelper.get_minimums(symbol)
        
        if not get_minimums:
            raise ValueError(f"Minimum trading details not found for symbol: {symbol}")
        
        step_size = float(str(get_minimums.get('stepSize', 1)))
        min_qty = float(str(get_minimums.get('minQty', 0)))
        
        # Adjust quantity to comply with step size
        adjusted_qty = math.floor(local_quantity / step_size) * step_size
        
        if adjusted_qty < min_qty:
            print(f"Quantity too small (min: {min_qty}).")
            return None, None

        # Calculate lots (now based on step size)
        lots = adjusted_qty / step_size
        
        if lots % 2 == 0:
            quant_to_sell = adjusted_qty / 2
        else:
            lots_to_sell = math.ceil(lots / 2)
            quant_to_sell = lots_to_sell * step_size
        
        # Ensure quant_to_sell respects step size again
        quant_to_sell = math.floor(quant_to_sell / step_size) * step_size
        remaining_qty = adjusted_qty - quant_to_sell
        
        print(f"Quantity to sell: {quant_to_sell}, Remaining: {remaining_qty}")
        return quant_to_sell, remaining_qty
    
    except Exception as e:
        print(f"Error in qty_to_exit: {e}")
        return None, None
def get_quantity(symbol: str, qty):
    """
    Returns the adjusted quantity based on Binance's lot size, min notional, and step size filters.

    :param symbol: Trading symbol (e.g., 'BTCUSDT')
    :param qty: Desired quantity to trade
    :return: Adjusted quantity that meets Binance's trading rules
    """
    
    # Fetch minimum trading values
    get_minimums = commonHelper.get_minimums(symbol)
    
    if not get_minimums:
        raise ValueError(f"Minimum trading details not found for symbol: {symbol}")
    
    # Fetch price
    price = get_ltp(symbol)
   

    # Convert values to Decimal for accuracy
    price = Decimal(str(price))
    qty = Decimal(str(qty))
    step_size = Decimal(str(get_minimums.get('stepSize', 1)))
    min_qty = Decimal(str(get_minimums.get('minQty', 0)))
    max_qty = Decimal(str(get_minimums.get('maxQty', float('inf'))))
    min_notional = Decimal(str(get_minimums.get('minNotional', 0)))

    # Ensure qty is within min/max qty limits
    qty = max(min(qty, max_qty), min_qty)
    
    # Ensure qty is a multiple of step size
    qty = (qty // step_size) * step_size  # Floor division to prevent exceeding limits
    
    # Ensure notional value meets min notional requirement
    notional = qty * price
    if notional < min_notional:
        qty = min_notional / price
        qty = (qty // step_size) * step_size  # Adjust to step size again

        if qty < min_qty:
            raise ValueError("Quantity is too small to meet Binance's min notional requirement.")

    # Return properly formatted quantity
    return float(qty.quantize(step_size, rounding=ROUND_DOWN))	
	
def get_min_trade_amount(symbol):
    #market = binance.market(symbol)
    #return float(market['limits']['amount']['min'])
	get_minimums = commonHelper.get_minimums(symbol)
	
	minQty = get_minimums.get('minQty')
	return minQty
	
def get_step_size(symbol):
    #market = binance.market(symbol)
    #return float(market['precision']['amount'])
	get_minimums = commonHelper.get_minimums(symbol)
	
	stepSize = get_minimums.get('stepSize')
	return stepSize
	
def get_min_notional(symbol):
    #market = binance.market(symbol)
    #return float(market['limits']['cost']['min'])
	get_minimums = commonHelper.get_minimums(symbol)
	
	minNotional = get_minimums.get('minNotional')
	return minNotional
	
def get_ltp(symbol):
    #ticker = binance.fetch_ticker(symbol)  
    ticker = binance2.get_current_price(symbol)
    last_price = ticker.get('price')
    if last_price is None:
        raise ValueError(f"Error: Could not fetch last price for {symbol}")
    return last_price  # Ensure return is properly aligned

def get_close_and_rsi(tradingsymbol, timeframe, delta,rsi_length):
        new_data_df= fetch_ohlcv(tradingsymbol)
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df)
        RSI, previous_rsi,rsi_df= calculate_last_rsi(new_data_df,rsi_length)
        print('get close and rsi:',datetime.datetime.now() )
                
        return C2,O2,RSI,previous_rsi,L2,p1_time
def last_2_candles_low_sl(sl_df):
       
        sl_df= sl_df.tail(2)
        first_row_low = sl_df.iloc[0]['low']
        second_row_low = sl_df.iloc[1]['low']
    
        #print("Low value of the first row:", first_row_low)
        #print("Low value of the second row:", second_row_low)
        #logFile.flush(); os.fsync(logFile.fileno());          
    
        lowest_low = sl_df['low'].min()
        #print('last 2 candles  low:',lowest_low )
        #logFile.flush(); os.fsync(logFile.fileno());          
    
        return lowest_low
		
def get_isl(df, order_exec_price):
    sl = last_2_candles_low_sl(df) 
    
    print('Last two candles low:', sl)
    print('Order execution price:', order_exec_price)

    Isl_percentage = os.getenv("Isl_percentage", "10%")  
    
    try:
        Isl_percentage = float(Isl_percentage.strip('%')) / 100
    except ValueError:
        raise ValueError(f"Invalid Isl_percentage value: {Isl_percentage}")

    # Calculate adjusted stop-loss (SL)
    if sl< (order_exec_price - (Isl_percentage *order_exec_price)):
            sl= order_exec_price - (Isl_percentage *order_exec_price)

    print('ISL:', sl)
   
    return sl  

def get_ltp_and_current_opt_open(tradingsymbol,opt_tf):
        ltp=get_ltp(tradingsymbol)
        print('Inside get_ltp_and_current_opt_open: ltp is:',ltp,opt_tf)
     
        new_data_df= fetch_ohlcv(tradingsymbol)
    
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df)
        print('returning ltp for trade time',datetime.datetime.now())
      
        
        return ltp,RO	
def calculate_last_bearish_candle(df):
        df = df.iloc[::-1]
    
        # find the first row with 'open' value greater than 'close'
        for row in df.itertuples():
            if row.open > row.close:
                result = row.low
                return result	

def get_ltp_and_current_opt_high(tradingsymbol,opt_tf):
        ltp=get_ltp(tradingsymbol)
        print('Inside get_ltp_and_current_opt_high: ltp is:',ltp,opt_tf)
          
        new_data_df= fetch_ohlcv(tradingsymbol)    
           
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df,True)        
        print('returning ltp for trade time',datetime.datetime.now())
             
        return ltp,H2
		
def get_last_bearish_candle(df,order_exec_price):
	sl= calculate_last_bearish_candle(df)
	print('last bearish candle low is',sl)
	Isl_percentage = os.getenv("Isl_percentage", "10%")  
	Isl_percentage = float(Isl_percentage.strip('%')) / 100
	if sl< (order_exec_price - (Isl_percentage *order_exec_price)):
		sl= order_exec_price - (Isl_percentage *order_exec_price) 
		
	print('sl',sl)
	return sl

def calculate_last_bearish_candle_with_rsi(df,lower_limit_value):
            df = df.iloc[::-1]
            for row in df.itertuples():
                if (row.open > row.close) and (row.rsi <lower_limit_value):
                    result = row.low
                    return result
					
def get_last_bearish_candle_with_rsi(df,order_exec_price,lower_limit_value):
	sl= calculate_last_bearish_candle_with_rsi(df,lower_limit_value)
	print('last bearish candle low is',sl)
	Isl_percentage = os.getenv("Isl_percentage", "10%")  
	Isl_percentage = float(Isl_percentage.strip('%')) / 100
	actual_ISL= order_exec_price - (Isl_percentage *order_exec_price)
	print('actual_ISL is',actual_ISL)
	if sl< actual_ISL:
		sl= actual_ISL 
	#print('sl is',sl)
	return sl	

def heikin_ashi(rel_df,new=False) :
        if not new:   
            df= heikin_ashi_df(rel_df)
        if new:
            df= heikin_ashi_df(rel_df,True)
        
        last_2_rows = df.tail(2)    
        C1 = last_2_rows.iloc[0]['close']
        C2 = last_2_rows.iloc[1]['close']
        O2 = last_2_rows.iloc[1]['open']
        H2 = last_2_rows.iloc[1]['high']
        L2=last_2_rows.iloc[1]['low']
        P1_time= last_2_rows.iloc[1]['date']
        last_row = last_2_rows.iloc[1][['open', 'close']]
        RO = round(last_row.mean(),2)
        #print('heikin_ashi time',datetime.datetime.now())
       
    
        return C1,C2,O2,H2,L2,RO,df,P1_time	
	
def get_data(symbol, timeframe, rsi_length):
                
        new_data_df= fetch_ohlcv(symbol)
        # if len(new_data_df.index) >0:
        C1,C2,O2,H2,L2,RO,hdf,p1_time = heikin_ashi(new_data_df)
        RSI,prev_rsi,rsi_df = calculate_last_rsi(new_data_df,rsi_length)
        #print('timeframe',timeframe)
        #print("get data time",datetime.datetime.now()) 
        print(C1,C2,O2,H2,L2,RO, RSI,prev_rsi,p1_time)
    
                  
        return C1,C2,O2,H2,L2,RSI,hdf,prev_rsi,p1_time,rsi_df
		


def EMA(df, base, target, period, alpha=False):
    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

    if alpha:
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()

    # Fix the warning
    df[target] = df[target].fillna(0)
    
    return df

def time_calc(opt_tf, entry):
    now = datetime.datetime.now().replace(microsecond=0)
    
    market_open = datetime.datetime(now.year, now.month, now.day, 9, 15, 0)
    start_time = market_open if entry else market_open + datetime.timedelta(seconds=4)
    
    intervals = {
        '1m': [start_time + datetime.timedelta(minutes=i) for i in range(375)],
        '2m': [start_time + datetime.timedelta(minutes=i*2) for i in range(188)],
        '3m': [start_time + datetime.timedelta(minutes=i*3) for i in range(125)],
        '4m': [start_time + datetime.timedelta(minutes=i*4) for i in range(94)],
        '5m': [start_time + datetime.timedelta(minutes=i*5) for i in range(75)],
        '10m': [start_time + datetime.timedelta(minutes=i*10) for i in range(38)],
        '15m': [start_time + datetime.timedelta(minutes=i*15) for i in range(25)],
        '30m': [start_time + datetime.timedelta(minutes=i*30) for i in range(13)],
        '1H': [start_time + datetime.timedelta(hours=i) for i in range(1, 7)],
        '2H': [start_time + datetime.timedelta(hours=i*2) for i in range(12)],
        '3H': [start_time + datetime.timedelta(hours=i*3) for i in range(8)]
    }

    if opt_tf not in intervals:
        return None  # Return None if the timeframe is invalid

    if market_open + datetime.timedelta(seconds=1) <= now <= market_open + datetime.timedelta(seconds=3):
        return intervals[opt_tf][1]

    target_times = intervals[opt_tf]
    
    idx = bisect.bisect_right(target_times, now)
    return target_times[idx] if idx < len(target_times) else None
		
def ATR(df, period, ohlc=['open', 'high', 'low', 'close']):
        atr = 'ATR_' + str(period)

        # Compute true range only if it is not computed and stored earlier in the df
        if not 'TR' in df.columns:
            df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
            df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
            df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

            df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

            df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

        # Compute EMA of true range using ATR formula after ignoring first row
        EMA(df, 'TR', atr, period, alpha=True)

        return df

def calculate_supertrend(df, period, multiplier):
    st = df.ta.supertrend(length=period, multiplier=multiplier)
    df["supertrend"] = st.iloc[:, 0]
    df["supertrend"] = df["supertrend"].fillna(0)  # Replace NaN with 0
    df["trend"] = np.where(df["close"] > df["supertrend"], "up", "down")

    latest_trend = df["trend"].iloc[-1]  # Correct indentation here
    return latest_trend


def data_analysis(tradingsymbol,opt_tf):
        
		 
    
        CC1,CC2,CO2,CH2,CL2,CRSI,cdf,prev_crsi,callp1_time,ce_rsi_df= get_data(tradingsymbol,opt_tf, RSI_LENGTH)
        
        call_supertrend_value=calculate_supertrend(cdf,int(length), int(factor))
        
	
        print("Candle data Extration completion time",datetime.datetime.now())
    
        return CC1,CC2,CO2,CH2,CL2,CRSI,prev_crsi,callp1_time,cdf,call_supertrend_value,ce_rsi_df


def create_trade_conditions(candle_condition, previous_candle, close_condition, condition, 
                          call_candle_condition, call_previous_candle, call_close_condition, call_condition,
                          put_candle_condition, put_previous_candle, 
                          put_close_condition, put_condition, ce_tradingsymbol, opt_tf, listcheck):
    
    while True:
        now = datetime.datetime.now()
        print('time is', now)                
        if 4 < now.second < 50:
            break
        time.sleep(1)

    CC1, CC2, CO2, CH2, CL2, CRSI, prev_crsi, callp1_time, cdf, call_supertrend_value, ce_rsi_df = data_analysis(ce_tradingsymbol, opt_tf)

    C1 = PC1 = CC1
    C2 = PC2 = CC2
    O2 = PO2 = CO2
    H2 = PH2 = CH2
    L2 = PL2 = CL2
    RSI=  PRSI = CRSI
    prev_rsi = prev_prsi = prev_crsi
    pdf = cdf
    trade_taken_time = putp1_time = callp1_time
    supertrend_value = put_supertrend_value = call_supertrend_value
    pe_rsi_df = fut_rsi_df = ce_rsi_df
    
    UL = RSI_UPPER
    LL = RSI_LOWER
    new_LL = PUT_UL
    new_UL = PUT_LL

    print('Returning all the values time:', datetime.datetime.now())
   
    candle_conditions = {
        'Higher Than Previous Candle Close and Above RSI Upper Limit': 'RSI>prev_rsi and RSI>UL',
        'Higher Than Previous Candle Close and Above RSI Lower Limit': 'RSI>prev_rsi and RSI>LL and RSI<UL',
        'Higher Than Previous Candle Close and Below RSI Lower Limit': 'RSI>prev_rsi and RSI<LL',
        'Lower than Previous Candle Close and Above the RSI Upper Limit': 'RSI<prev_rsi and RSI>UL',
        'Lower than Previous Candle Close and Below the RSI Upper Limit': 'RSI<prev_rsi and RSI>LL and RSI<UL',
        'Lower than Previous Candle Close and Below the RSI Lower Limit': 'RSI<prev_rsi and RSI<LL'}

    call_candle_conditions = {
        'Higher Than Previous Candle Close and Above RSI Upper Limit': 'CRSI>prev_crsi and CRSI>UL',
        'Higher Than Previous Candle Close and Above RSI Lower Limit': 'CRSI>prev_crsi and CRSI>LL and CRSI<UL',
        'Higher Than Previous Candle Close and Below RSI Lower Limit': 'CRSI>prev_crsi and CRSI<LL',
        'Lower than Previous Candle Close and Above the RSI Upper Limit': 'CRSI<prev_crsi and CRSI>UL',
        'Lower than Previous Candle Close and Below the RSI Upper Limit': 'CRSI<prev_crsi and CRSI>LL and CRSI<UL',
        'Lower than Previous Candle Close and Below the RSI Lower Limit': 'CRSI<prev_crsi and CRSI<LL'}
        
    put_candle_conditions = {
        'Lower than Previous Candle Close and Above the RSI Upper Limit': 'prev_prsi>PRSI and PRSI>UL',
        'Lower than Previous Candle Close and Below the RSI Upper Limit': 'prev_prsi>PRSI and PRSI<UL and PRSI>LL',
        'Lower than Previous Candle Close and Below the RSI Lower Limit': 'prev_prsi>PRSI and PRSI<LL',
        'Higher Than Previous Candle Close and Below RSI Lower Limit': 'prev_prsi<PRSI and PRSI<LL',
        'Higher Than Previous Candle Close and Above RSI Lower Limit': 'prev_prsi<PRSI and PRSI<UL and PRSI>LL',
        'Higher Than Previous Candle Close and Above RSI Upper Limit': 'prev_prsi<PRSI and PRSI>UL'}

    # Define the conditions for each dropdown selection for "previous candle"
    previous_candle_dict = {
        'Open': 'O2',
        'High': 'H2',
        'Low': 'L2',
        'Close': 'C2'
    }
    call_previous_candle_dict = {
        'Open': 'CO2',
        'High': 'CH2',
        'Low': 'CL2',
        'Close': 'CC2'
    }
    put_previous_candle_dict = {
        'Open': 'PO2',
        'High': 'PH2',
        'Low': 'PL2',
        'Close': 'PC2'
    }

    # Define the conditions for each dropdown selection for "condition"
    condition_dict = {
        'Open': 'O2',
        'High': 'H2',
        'Low': 'L2',
        'Close': 'C2'
    }
    call_condition_dict = {
        'Open': 'CO2',
        'High': 'CH2',
        'Low': 'CL2',
        'Close': 'CC2'
    }
    put_condition_dict = {
        'Open': 'PO2',
        'High': 'PH2',
        'Low': 'PL2',
        'Close': 'PC2'
    }

    # Define the conditions for each dropdown selection for "close condition"
    close_conditions = {
        'Lower Than or Equal To': '<=',
        'Higher Than or Equal to': '>=',
        'Lower Than': '<',
        'Higher Than': '>',
        'Equal To': '=='
    }

    # Combine the selected conditions
    combined_condition = f"{previous_candle_dict[previous_candle]} {close_conditions[close_condition]} {condition_dict[condition]}"
    call_combined_condition = f"{call_previous_candle_dict[call_previous_candle]} {close_conditions[call_close_condition]} {call_condition_dict[call_condition]}"
    put_combined_condition = f"{put_previous_candle_dict[put_previous_candle]} {close_conditions[put_close_condition]} {put_condition_dict[put_condition]}"
    
    if CHART_REFERENCE == 'future_chart':
        print("Selected Future Chart")
        if THIRD_CHART_REFERENCE == 'yes':
            call_condition_str = f"({candle_conditions[candle_condition]} and {combined_condition}) and " \
                                f"({call_candle_conditions[call_candle_condition]} and {call_combined_condition}) and " \
                                f"({put_candle_conditions[put_candle_condition]} and {put_combined_condition})"
    
            put_condition_str = call_condition_str.replace('>', 'temp').replace('<', '>').replace('temp', '<')
            put_condition_str = put_condition_str.replace('UL', 'temp').replace('LL', 'UL').replace('temp', 'LL')
            conditions = put_condition_str.split(') and (')
            print(conditions)

            # Replace the UL and LL in the first two conditions
            conditions[0] = conditions[0].replace('LL', 'new_LL').replace('UL', 'new_UL')
            conditions[1] = conditions[1].replace('LL', 'new_LL').replace('UL', 'new_UL')
            print(conditions[0], conditions[1])

            # Reconstruct the put condition string with the updated first two conditions
            put_condition_str = f"({') and ('.join(conditions)})"
            
            print("call condition string which includes third chart as well", call_condition_str)
            print("put condition string which includes third chart as well", put_condition_str)

        elif THIRD_CHART_REFERENCE == 'no':
            call_condition_str = f"({candle_conditions[candle_condition]} and {combined_condition}) and " \
                                f"({call_candle_conditions[call_candle_condition]} and {call_combined_condition})"
            
            put_condition_str = f"({candle_conditions[candle_condition]} and {combined_condition}) and " \
                               f"({put_candle_conditions[put_candle_condition]} and {put_combined_condition})"
            
            put_condition_str = put_condition_str.replace('>', 'temp').replace('<', '>').replace('temp', '<')
            put_condition_str = put_condition_str.replace('UL', 'temp').replace('LL', 'UL').replace('temp', 'LL')
            conditions = put_condition_str.split(') and (')
            print("conditions are", conditions)

            # Replace the UL and LL in the first two conditions
            conditions[0] = conditions[0].replace('LL', 'new_LL').replace('UL', 'new_UL')
            print(conditions[0])
            
            # Reconstruct the put condition string with the updated first two conditions
            put_condition_str = f"({') and ('.join(conditions)})"

            print("call condition string which doesnt include third chart", call_condition_str)
            print("put condition string which doesnt include third chart", put_condition_str)

    elif CHART_REFERENCE == 'option_chart':
        print("Selected Option Chart")
        if THIRD_CHART_REFERENCE == 'yes':
            call_condition_str = f"({call_candle_conditions[call_candle_condition]} and {call_combined_condition}) and " \
                                f"({put_candle_conditions[put_candle_condition]} and {put_combined_condition})"
    
            put_condition_str = call_condition_str.replace('>', 'temp').replace('<', '>').replace('temp', '<')
            put_condition_str = put_condition_str.replace('UL', 'temp').replace('LL', 'UL').replace('temp', 'LL')
            conditions = put_condition_str.split(') and (')
            print(conditions)

            # Replace the UL and LL in the first two conditions
            conditions[0] = conditions[0].replace('LL', 'new_LL').replace('UL', 'new_UL')
            conditions[1] = conditions[1].replace('LL', 'new_LL').replace('UL', 'new_UL')
            print(conditions[0], conditions[1])

            # Reconstruct the put condition string with the updated first two conditions
            put_condition_str = f"({') and ('.join(conditions)})"
            
            print("call condition string which includes third chart as well", call_condition_str)
            print("put condition string which includes third chart as well", put_condition_str)

        elif THIRD_CHART_REFERENCE == 'no':
            call_condition_str = f"({call_candle_conditions[call_candle_condition]} and {call_combined_condition})"
            
            put_condition_str = f"({put_candle_conditions[put_candle_condition]} and {put_combined_condition})"
            
            put_condition_str = put_condition_str.replace('>', 'temp').replace('<', '>').replace('temp', '<')
            put_condition_str = put_condition_str.replace('UL', 'temp').replace('LL', 'UL').replace('temp', 'LL')
            conditions = put_condition_str.split(') and (')
            print("conditions are", conditions)

            # Replace the UL and LL in the first two conditions
            conditions[0] = conditions[0].replace('LL', 'new_LL').replace('UL', 'new_UL')
            print(conditions[0])
            
            # Reconstruct the put condition string with the updated first two conditions
            put_condition_str = f"({') and ('.join(conditions)})"

            print("call condition string which doesnt include third chart", call_condition_str)
            print("put condition string which doesnt include third chart", put_condition_str)

    print("checking conditions")        
    print_str = "" #f"Thread {threading.current_thread()}  \n"
    print_str += f"Current Time: {datetime.datetime.now()}  \n"            
    print_str += f"1. P2 call candle RSI: {prev_crsi} | 2. P1 call candle close: {CC2} | 3. P1 call open: {CO2} | 4. P1 call high: {CH2} | 5. P1 call low: {CL2} | 6. P1 call RSI: {round(CRSI,2)} Candle Time:{callp1_time}\n"
    print_str += f"7. P2 put candle RSI: {prev_prsi} | 8. P1 put candle close: {PC2} | 9. P1 put open: {PO2} | 10. P1 put high: {PH2} | 11. P1 put low: {PL2} | 12. P1 put RSI: {round(PRSI,2)} Candle Time:{putp1_time}\n"
    print_str += f"13. P2 fut candle RSI: {prev_rsi} | 14. P1 fut candle close: {C2} | 15. P1 fut open: {O2} | 16. P1 fut high: {H2} | 17. P1 fut low:  {L2} | 18. P1 fut RSI: {round(RSI,2)} Candle Time:{trade_taken_time} |"
    print('\nCondition checking!:', print_str, 'UL-->:', RSI_UPPER, 'LL-->:', RSI_LOWER)
    print('CALL entry Conditions:', (call_condition_str), eval(call_condition_str))
    print('PUT entry Conditions:', (put_condition_str), eval(put_condition_str))
    
    if CHART_REFERENCE == 'future_chart':
        if BUY_ONLY_CALL_OR_PUT == "both":     
            #time.sleep(60)
            if SUPERTREND == 'yes':
                previous_put_condition_met = eval(put_condition_str)
                previous_call_condition_met = eval(call_condition_str)
                if previous_call_condition_met and supertrend_value == 'up':
                    print("call condition met and super trend value is up so taking call trade")
                    call_condition_met = True
                    put_condition_met = None
                elif previous_put_condition_met and supertrend_value == 'down':
                    print("put condition met and super trend value is down so taking put trade")
                    put_condition_met = True
                    call_condition_met = None
                else:
                    if previous_call_condition_met or previous_put_condition_met:
                        print("call or put conditions met but super trend conditions havent been met ")
                        put_condition_met = None
                        call_condition_met = None
                    else:
                        print("call and put condition was not satisfied in the first place")
                        put_condition_met = None
                        call_condition_met = None
            else:
                print("supertrend is not considered")
                put_condition_met = eval(put_condition_str)
                call_condition_met = eval(call_condition_str)
            
            print(f" call condition {call_condition_met} and put condition {put_condition_met}")
            if call_condition_met:
                print_str = "CALL CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                print_str += f"Current Time: {datetime.datetime.now()}  \n"
                print('CALL entry Conditions:', (call_condition_str), eval(call_condition_str))
                print("Call conditions met and starting to take trade")
                
                signal = True
                if signal:
                    t1 = cdf['date'].iloc[-1]
                    t2 = cdf['date'].iloc[-2]
                    if not (t1 in listcheck and t2 in listcheck):
                        print('t1 and t2 not in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
                        logFile.flush()
                        os.fsync(logFile.fileno())

                        listcheck.clear()
                        listcheck.append(t1)
                        listcheck.append(t2)
                        
                        s = True
                        put_condition_met = None
                    else:
                        print("trade had benn already taken for the entry candle so not taking trade again")
                        print('t1 and t2 is already in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
                        call_condition_met = None
                        put_condition_met = None

            elif put_condition_met:
                print_str = "PUT CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                print_str += f"Current Time: {datetime.datetime.now()}  \n"                    
                print('PUT entry Conditions:', (put_condition_str), eval(put_condition_str))
                print("Put conditions met and starting to take trade")
                
                signal = True
                if signal:
                    t1 = pdf['date'].iloc[-1]
                    t2 = pdf['date'].iloc[-2]
                    if not (t1 in listcheck and t2 in listcheck):
                        listcheck.clear()
                        listcheck.append(t1)
                        listcheck.append(t2)
                        
                        call_condition_met = None
                        put_condition_met = True
                        print("Since there is no existing call trades, going to take put trade")
                    else:
                        call_condition_met = None
                        put_condition_met = None 
                        print("trade had benn already taken for the entry candle so not taking trade again")
                        print('t1 and t2 is already in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
            
            return call_condition_met, put_condition_met, cdf, pdf, listcheck, supertrend_value, CC2, CO2, CH2, CL2, PC2, PO2, PH2, PL2, ce_rsi_df, pe_rsi_df, fut_rsi_df
            
        elif BUY_ONLY_CALL_OR_PUT == "call":
            if SUPERTREND == 'yes':
                previous_call_condition_met = eval(call_condition_str)
                if previous_call_condition_met and supertrend_value == 'up':
                    print("call condition met and super trend value is up so taking call trade")
                    call_condition_met = True
                    put_condition_met = None
                else:
                    if previous_call_condition_met:
                        print("call condition met but super trend value is not up so not  taking call trade")
                        put_condition_met = None
                        call_condition_met = None
                    else:
                        print("call condition was not satisfied in the first place")
                        put_condition_met = None
                        call_condition_met = None
            else:
                print("super trend is not considered")
                call_condition_met = eval(call_condition_str)               
                put_condition_met = None

            print(f" only call should be taken. call condition {call_condition_met} and put condition {put_condition_met}")
            if call_condition_met:
                print_str = "CALL CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                print_str += f"Current Time: {datetime.datetime.now()}  \n"
                print('CALL entry Conditions:', (call_condition_str), eval(call_condition_str))
                print("Call conditions met and starting to take trade")
                
                signal = True
                if signal:
                    t1 = cdf['date'].iloc[-1]
                    t2 = cdf['date'].iloc[-2]
                    if not (t1 in listcheck and t2 in listcheck):
                        listcheck.clear()
                        listcheck.append(t1)
                        listcheck.append(t2)
                        call_condition_met = True
                        put_condition_met = None
                        print("Since there is no existing put trades, going to take call trade")
                    else:
                        call_condition_met = None
                        put_condition_met = None
                        print("trade had been already taken for the entry candle so not taking trade again")
                        print('t1 and t2 is already in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
            
            return call_condition_met, put_condition_met, cdf, pdf, listcheck, supertrend_value, CC2, CO2, CH2, CL2, PC2, PO2, PH2, PL2, ce_rsi_df, pe_rsi_df, fut_rsi_df

        elif BUY_ONLY_CALL_OR_PUT == "put":
            if SUPERTREND == 'yes':
                previous_put_condition_met = eval(put_condition_str)
                if previous_put_condition_met and supertrend_value == 'down':
                    print("put condition met and super trend value is down so taking put trade")
                    put_condition_met = True
                    call_condition_met = None
                else:
                    if previous_put_condition_met:
                        print("put condition met but super trend value is not down so taking not taking put trade")
                        put_condition_met = None
                        call_condition_met = None
                    else:
                        print("put condition was not satisfied in the first place")
                        put_condition_met = None
                        call_condition_met = None
            else:
                print("super trend is not considered")
                put_condition_met = eval(put_condition_str)
                call_condition_met = None

            print(f" only put should be taken. call condition {call_condition_met} and put condition {put_condition_met}")
            if put_condition_met:
                print_str = "PUT CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                print_str += f"Current Time: {datetime.datetime.now()}  \n"
                print('PUT entry Conditions:', (put_condition_str), eval(put_condition_str))
                print("Put conditions met and starting to take trade")
                
                signal = True
                if signal:
                    t1 = pdf['date'].iloc[-1]
                    t2 = pdf['date'].iloc[-2]
                    if not (t1 in listcheck and t2 in listcheck):
                        listcheck.clear()
                        listcheck.append(t1)
                        listcheck.append(t2)
                        call_condition_met = None
                        put_condition_met = True
                        print("Since there is no existing call trades, going to take put trade")
                    else:
                        call_condition_met = None
                        put_condition_met = None 
                        print("trade had been already taken for the entry candle so not taking trade again")
                        print('t1 and t2 is already in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
        
            print(call_condition_met, put_condition_met)
            return call_condition_met, put_condition_met, cdf, pdf, listcheck, supertrend_value, CC2, CO2, CH2, CL2, PC2, PO2, PH2, PL2, ce_rsi_df, pe_rsi_df, fut_rsi_df

    elif CHART_REFERENCE == 'option_chart':
        if BUY_ONLY_CALL_OR_PUT == "both":     
            if SUPERTREND == 'yes':
                previous_put_condition_met = eval(put_condition_str)
                previous_call_condition_met = eval(call_condition_str)
                print("call st value is", call_supertrend_value)
                print("put st value is", put_supertrend_value)
                if previous_call_condition_met and call_supertrend_value == 'up':
                    print("call condition met and super trend value is up so taking call trade")
                    call_condition_met = True
                    put_condition_met = None
                elif previous_put_condition_met and put_supertrend_value == 'up':
                    print("put condition met and super trend value is down so taking put trade")
                    put_condition_met = True
                    call_condition_met = None
                else:
                    if previous_call_condition_met or previous_put_condition_met:
                        print("call or put conditions met but super trend conditions havent been met ")
                        put_condition_met = None
                        call_condition_met = None
                    else:
                        print("call and put condition was not satisfied in the first place")
                        put_condition_met = None
                        call_condition_met = None
            else:
                print("supertrend is not considered")
                put_condition_met = eval(put_condition_str)
                call_condition_met = eval(call_condition_str)
            
            print(f" call condition {call_condition_met} and put condition {put_condition_met}")
            if call_condition_met:
                print_str = "CALL CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                print_str += f"Current Time: {datetime.datetime.now()}  \n"
                print('CALL entry Conditions:', (call_condition_str), eval(call_condition_str))
                print("Call conditions met and starting to take trade")
                
                signal = True
                if signal:
                    t1 = cdf['date'].iloc[-1]
                    t2 = cdf['date'].iloc[-2]
                    if not (t1 in listcheck and t2 in listcheck):
                        print('t1 and t2 not in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
                        
                        listcheck.clear()
                        listcheck.append(t1)
                        listcheck.append(t2)
                        
                        call_condition_met = True
                        put_condition_met = None
                    else:
                        print("trade had benn already taken for the entry candle so not taking trade again")
                        print('t1 and t2 is already in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
                        call_condition_met = None
                        put_condition_met = None

            elif put_condition_met:
                print_str = "PUT CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                print_str += f"Current Time: {datetime.datetime.now()}  \n"                    
                print('PUT entry Conditions:', (put_condition_str), eval(put_condition_str))
                print("Put conditions met and starting to take trade")
                
                signal = True
                if signal:
                    t1 = pdf['date'].iloc[-1]
                    t2 = pdf['date'].iloc[-2]
                    if not (t1 in listcheck and t2 in listcheck):
                        listcheck.clear()
                        listcheck.append(t1)
                        listcheck.append(t2)
                        
                        call_condition_met = None
                        put_condition_met = True
                        print("Since there is no existing call trades, going to take put trade")
                    else:
                        call_condition_met = None
                        put_condition_met = None 
                        print("trade had benn already taken for the entry candle so not taking trade again")
                        print('t1 and t2 is already in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
            
            return call_condition_met, put_condition_met, cdf, pdf, listcheck, supertrend_value, CC2, CO2, CH2, CL2, PC2, PO2, PH2, PL2, ce_rsi_df, pe_rsi_df, fut_rsi_df
            
        elif BUY_ONLY_CALL_OR_PUT == "call":
            if SUPERTREND == 'yes':
                print("call st value is", call_supertrend_value)
                print("put st value is", put_supertrend_value)
                
                previous_call_condition_met = eval(call_condition_str)
                if previous_call_condition_met and call_supertrend_value == 'up':
                    print("call condition met and super trend value is up so taking call trade")
                    call_condition_met = True
                    put_condition_met = None
                else:
                    if previous_call_condition_met:
                        print("call condition met but super trend value is not up so not  taking call trade")
                        put_condition_met = None
                        call_condition_met = None
                    else:
                        print("call condition was not satisfied in the first place")
                        put_condition_met = None
                        call_condition_met = None
            else:
                print("super trend is not considered")
                call_condition_met = eval(call_condition_str)               
                put_condition_met = None

            print(f" only call should be taken. call condition {call_condition_met} and put condition {put_condition_met}")
            if call_condition_met:
                print_str = "CALL CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                print_str += f"Current Time: {datetime.datetime.now()}  \n"
                print('CALL entry Conditions:', (call_condition_str), eval(call_condition_str))
                print("Call conditions met and starting to take trade")
                
                signal = True
                if signal:
                    t1 = cdf['date'].iloc[-1]
                    t2 = cdf['date'].iloc[-2]
                    if not (t1 in listcheck and t2 in listcheck):
                        listcheck.clear()
                        listcheck.append(t1)
                        listcheck.append(t2)
                        call_condition_met = True
                        put_condition_met = None
                        print("Since there is no existing put trades, going to take call trade")
                    else:
                        call_condition_met = None
                        put_condition_met = None
                        print("trade had been already taken for the entry candle so not taking trade again")
                        print('t1 and t2 is already in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
            
            return call_condition_met, put_condition_met, cdf, pdf, listcheck, supertrend_value, CC2, CO2, CH2, CL2, PC2, PO2, PH2, PL2, ce_rsi_df, pe_rsi_df, fut_rsi_df

        elif BUY_ONLY_CALL_OR_PUT == "put":
            print(" curb your enthusiasm call st value is", call_supertrend_value)
            print("put st value is", put_supertrend_value)
            if SUPERTREND == 'yes':
                previous_put_condition_met = eval(put_condition_str)
                if previous_put_condition_met and put_supertrend_value == 'up':
                    print("put condition met and super trend value is down so taking put trade")
                    put_condition_met = True
                    call_condition_met = None
                else:
                    if previous_put_condition_met:
                        print("put condition met but super trend value is not down so taking not taking put trade")
                        put_condition_met = None
                        call_condition_met = None
                    else:
                        print("put condition was not satisfied in the first place")
                        put_condition_met = None
                        call_condition_met = None
            else:
                print("super trend is not considered")
                put_condition_met = eval(put_condition_str)
                call_condition_met = None

            print(f" only put should be taken. call condition {call_condition_met} and put condition {put_condition_met}")
            if put_condition_met:
                print_str = "PUT CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                print_str += f"Current Time: {datetime.datetime.now()}  \n"
                print('PUT entry Conditions:', (put_condition_str), eval(put_condition_str))
                print("Put conditions met and starting to take trade")
                
                signal = True
                if signal:
                    t1 = pdf['date'].iloc[-1]
                    t2 = pdf['date'].iloc[-2]
                    if not (t1 in listcheck and t2 in listcheck):
                        listcheck.clear()
                        listcheck.append(t1)
                        listcheck.append(t2)
                        call_condition_met = None
                        put_condition_met = True
                        print("Since there is no existing call trades, going to take put trade")
                    else:
                        call_condition_met = None
                        put_condition_met = None 
                        print("trade had been already taken for the entry candle so not taking trade again")
                        print('t1 and t2 is already in listcheck')
                        print('t1 is', t1, 't2 is', t2, 'listcheck is', listcheck)
        
            print(call_condition_met, put_condition_met)
            return call_condition_met, put_condition_met, cdf, pdf, listcheck, supertrend_value, CC2, CO2, CH2, CL2, PC2, PO2, PH2, PL2, ce_rsi_df, pe_rsi_df, fut_rsi_df

def check_tsl(prev_rsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity):
        
        if (((call_trade_taken and prev_rsi>rsi and (orsi < RSI_UPPER 
                )) or (put_trade_taken and rsi>prev_rsi
                and (orsi < RSI_UPPER)))
                and (not entered_condition) 
                and ts1_condition_satisfied 
                and (TSL1== 'Exit at market price and half trailing with bearish candle low sl' or TSL1== 'Exit half and remaining at bearish candle low')) :
        
        
                                           
            print(f"given {TSL1} condition satisfied for the candle time of {last_candle_time}")
            

            entered_condition = True
            all_exited, sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  

        
        elif ( ((call_trade_taken and prev_rsi>rsi
                and  orsi< RSI_LOWER)
                or (put_trade_taken and rsi>prev_rsi
                and  orsi< RSI_LOWER))
                and (not entered_condition)  and (TSL2== 'Exit at market price and half trailing with bearish candle low sl' or TSL2== 'Exit half and remaining at bearish candle low')):
        # if True:  
                print(f"given {TSL2} condition satisfied for the candle time of {last_candle_time}")                                                        
               
                
                entered_condition = True  
                all_exited,sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,logFile,i,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  

        else:
            print("TSL condition not satisfies so not taking any trades")
            

        return all_exited,sl_order_id,local_quantity,entered_condition,isl_value

def checking_condition(trade, ssl, last_candle_close, order_exec_price, name, opt_tf, sl_order_id, all_exited, local_quantity, tslmovement, trailing_sl_movement, trailing_sl_target, single_candle_tsl, single_candle_sl):
    print("entered into exit at market checking conditionm")

    if last_candle_close < order_exec_price:  # exit all quantity logic
        print("p1_candle_low < order_exec_price ", 'last low is', last_candle_close, 'order_executed price is', order_exec_price)
        print(" condition satisfied closing the trade")

        order_sl_id = binance2.modify_mkt_order(sl_order_id, local_quantity, name)
        # status = orderplacement.get_orderstatus(exit_all, logFile)
        order_sl_id = order_sl_id["orderId"]
        status = binance2.get_order_status(name, order_sl_id)  # This is the line you referenced
        
        if status == 'FILLED':
            all_exited = True
            print("exited all")

            local_quantity = 0
            return all_exited, sl_order_id, tslmovement, trailing_sl_movement, trailing_sl_target, single_candle_tsl, single_candle_sl, local_quantity, ssl

    if True and not all_exited:
        sl_order_id, tslmovement, trailing_sl_movement, trailing_sl_target, single_candle_tsl, single_candle_sl, ssl, all_exited, isl_value = single_candle_condition_checking(isl_value, trade, all_exited, ssl, local_quantity, name, opt_tf, sl_order_id, tslmovement, trailing_sl_movement, trailing_sl_target, single_candle_tsl, single_candle_sl)
        o_ltp = get_ltp(name)
        ltpp, high = get_ltp_and_current_opt_high(name, opt_tf)

        all_exited, local_quantity, first_target, second_target, sl_order_id = target_checking(high, first_target, second_target, Order_Identifier, isl_value, name, trade, o_ltp, order_exec_price, all_exited, sl_order_id, local_quantity)

    return all_exited, sl_order_id, tslmovement, trailing_sl_movement, trailing_sl_target, single_candle_tsl, single_candle_sl, local_quantity, ssl

def target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity):
        
        if TARGET_SPLIT=="enable":
            print("Target split is enabled")
            first_target_trailing=False
            second_target_trailing=False
            second_sl=False
            if first_target and not first_target_trailing:
                first_targett= FIRST_TARGET
                target= order_exec_price+ (first_targett*order_exec_price)
                print("First target is",target)
                
                

                if o_ltp> target or high> target:
                    print("o_ltp is",o_ltp)
                    print("high is",high)
                    print("first target is",target)

                    print("First target has been achieved")
                    first_qty,local_quantity=qty_to_exit(name,local_quantity)
                    first_target_percent= FIRST_TARGET_TRAILING
                    print("first target percent is", first_target_percent)
                    total_point=target*first_target_percent
                    print("total_point", total_point)

                    first_target_sl= target-total_point
                    print("first target sl is",first_target_sl)
                    sl_price= round(float(first_target_sl),1)
                    isl_value=sl_price
                    price= round(float(first_target_sl-0.1),1) 
                    sl_order_id,all_exited= binance2.modify_stop_loss(name, sl_order_id, sl_price, price, first_qty,all_exited)                    
                    first_target_trailing =True
                    first_target_trailing_point= target + total_point
                    print("first target trailing point is",first_target_trailing_point)
                    
                    print("sl has been modified for half quantity")
                    
                    
                    first_target=False
                                
                    if all_exited:
                        second_target=True
                        all_exited=False
                        first_target_trailing=False
                        first_target=False
                        second_sl=True
            
            if first_target_trailing:
                status = binance2.get_order_status(name,sl_order_id)
                print("first targetr order status is not completed entering into while loo[p]")
                first_target_percent= FIRST_TARGET_TRAILING
                while status !="COMPLETE":
                    o_ltp=get_ltp(name)
                    if o_ltp > first_target_trailing_point:
                        
                        isl_value=isl_value+total_point
                        sl_price= round(float(isl_value),1)
                        price= round(float(isl_value-0.1),1) 
                        sl_order_id,all_exited= binance2.modify_stop_loss(name, sl_order_id, sl_price, price, first_qty,all_exited)
                        if all_exited:
                            second_target=True
                            all_exited=False
                            first_target_trailing=False
                            first_target=False
                            break

                        first_target_trailing_point=first_target_trailing_point+total_point
                    
                    status = binance2.get_order_status(name,sl_order_id)
                    
                print("first targetr order status is completed entering into second target")
                second_target=True
                all_exited=False
                first_target_trailing=False
                first_target=False
                second_sl=True


            if second_target:
                if second_sl:
                                        
                    sl_price= round(float(order_exec_price),1)
                    price = round(float(order_exec_price - 0.1), 1)
                    order = binance2.create_slm_order('sell', name, local_quantity, sl_price, price)
                    sl_order_id = order["orderId"]
                    all_exited = True
                            
                    print("second Sl has been placed for remaining qty in order exec price")
                    second_sl=False
                    isl_value=order_exec_price
            
                second_targett= SECOND_TARGET
                target= order_exec_price+ (second_targett*order_exec_price)
                

                print("second target is",target)
                
                if o_ltp> target or high > target :
                    print("second target has been achieved ")
                    print("o_ltp is",o_ltp)
                    print("high is",high)
                    print("second target is",target)
                    second_target_percent= SECOND_TARGET_TRAILING
                    print("second target percent is", second_target_percent)
                    total_point=target*second_target_percent
                    print("second total_point", total_point)
                    second_target_sl= target-total_point
                    print("second target sl is",second_target_sl)
                    sl_price= round(float(second_target_sl),1)
                    isl_value=sl_price
                    price= round(float(second_target_sl-0.1),1) 
                    sl_order_id,all_exited= binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)                    
                    second_target_trailing =True
                    second_target_trailing_point= target + (target*SECOND_TARGET_TRAILING)
                    print("second target trailing point is",second_target_trailing_point)
                    
                    print("sl has been modified for half quantity")
                    
                    
                    
                    second_target=False
            
                    if all_exited:
                        second_target_trailing=False
                        all_exited=True
                        
            
            if second_target_trailing:
                status = binance2.get_order_status(name, sl_order_id)
                print("second targetr order status is not completed entering into while loo[p]")
                second_target_percent= SECOND_TARGET_TRAILING
                while status !="COMPLETE":
                    o_ltp=get_ltp(name)
                    if o_ltp > second_target_trailing_point:
                        
                        isl_value=isl_value+total_point
                        sl_price= round(float(isl_value),1)
                        price= round(float(isl_value-0.1),1) 
                        sl_order_id, all_exited = binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
                        if all_exited:
                            
                            all_exited=True
                            
                            break

                        second_target_trailing_point=second_target_trailing_point+total_point
                    
                    status = binance2.get_order_status(name, sl_order_id)
                    
                print("second targetr order status is completed entering into second target")
              
                all_exited=True
                local_quantity=0

                return all_exited,local_quantity,first_target,second_target,sl_order_id

        
        elif TARGET_SPLIT=="disable":
            print("Target split is disabled") 
            target_percent= TARGET
            target= order_exec_price+ (target_percent*order_exec_price)

            print('target is',target)
           

            if o_ltp> target or high > target:  ##logic to exit the full quantities
                oid= binance2.modify_mkt_order(sl_order_id, local_quantity, name)
                time.sleep(5)
                #status= orderplacement.get_orderstatus(oid,logFile)
                status = binance2.get_order_status(name, oid)
            
                if status=='FILLED':
                    all_exited=True
                    
                    local_quantity=0

                else:
                    print("status is not complete",status)
            
            return all_exited,local_quantity,first_target,second_target,sl_order_id
            
        return all_exited,local_quantity,first_target,second_target,sl_order_id

def full_trailing_checking(name,trade,opt_low,isl_value,sl_order_id,all_exited,local_quantity):
        if opt_low > isl_value:
            sl_price= round(float(opt_low),1)
            price= round(float(opt_low-0.1),1) 
            sl_order_id,all_exited= binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
            
            

            isl_value=opt_low
            print("Shifting isl/tsl value")
            print("New tsl is",isl_value)
           
        return sl_order_id,isl_value,all_exited
		
def bullish_condition(bull_condt,order_executed_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name, timeframe,rsi_length):
        new_data_df= fetch_ohlcv(name)
        df= heikin_ashi_df(new_data_df)
        bullish_rows = df.tail(3)
        bullish_rows = bullish_rows.reset_index(drop=True)
        print("bullish condition for same candle is", bull_condt)
        print(bullish_rows)
        are_bullish = all(bullish_rows['close'] > bullish_rows['open'])
        print("Three continous bullish candle satisfied",are_bullish)
        
        if are_bullish:
            print("Three previous candles are continously bullish")
            print(f"p1_close: {bullish_rows.loc[2, 'close']}")
            print(f"p1_open: {bullish_rows.loc[2, 'open']}")
            print(f"p2_close: {bullish_rows.loc[1, 'close']}")
            print(f"p2_open: {bullish_rows.loc[1, 'open']}")
            
            print(f"p3_close: {bullish_rows.loc[0, 'close']}")
            print(f"p3_open: {bullish_rows.loc[0, 'open']}")

            row1 = bullish_rows.iloc[0][BULLISH_CONDITION]
            print("p3 candle selected value is",bullish_rows.iloc[0][BULLISH_CONDITION])
            row2 = bullish_rows.iloc[1][BULLISH_CONDITION]
            print("p2 candle selected value is",bullish_rows.iloc[1][BULLISH_CONDITION])
            
            row3 = bullish_rows.iloc[2][BULLISH_CONDITION]
            print("p1 candle selected value is",bullish_rows.iloc[2][BULLISH_CONDITION])


            if row1 < order_executed_price or row2 < order_executed_price or row3 < order_executed_price:
                print("Chosen value is lesser than order executed price so not executing bullish condition logic")
                return sl_order_id,isl_value,all_exited,bull_condt
            else:
                lowest_low = bullish_rows['low'].min()
                print("Chosen value is greater than order executed price so executing bullish condition logic")
                
                print(f"The lowest value in the 'low' column is: {lowest_low}")
                sl_price= round(float(lowest_low),1)
                print("Three bullish candle logic sl price is",sl_price)
                price= round(float(lowest_low-0.1),1) 
                print("bullish conditionm value is",bull_condt)
                if ((bull_condt) and (isl_value<lowest_low)):
                    bull_condt=False
                    sl_order_id, all_exited = binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
                    print("Shifting isl/tsl value")
                    isl_value=lowest_low

                              
               
                return sl_order_id,isl_value,all_exited,bull_condt

        else:
            print("Three previous candles aren't continously bullish")
            print(f"p1_close: {bullish_rows.loc[0, 'close']}")
            print(f"p1_open: {bullish_rows.loc[0, 'open']}")
            print(f"p2_close: {bullish_rows.loc[1, 'close']}")
            print(f"p2_open: {bullish_rows.loc[1, 'open']}")
            print(f"p3_close: {bullish_rows.loc[2, 'close']}")
            print(f"p3_open: {bullish_rows.loc[2, 'open']}")
            return sl_order_id,isl_value,all_exited,bull_condt		
def single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl):
        print("entered into single candle condition checking")
        print("tsl movement is", tslmovement)
        
    
        no_of_times= SINGLE_CANDLE_CONDITION  
        o_ltp,current_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
        single_candle_target= (no_of_times*current_candle_open) + current_candle_open
        print('no of times:',no_of_times, 'o_ltp;',o_ltp, 'single candle target', single_candle_target, 'current candle open',current_candle_open)
        
       
        ###target checking###
        if o_ltp> single_candle_target:
            print("single candle target achieved")
            
            selected_percent= TSLOFSCC   
            single_percent_sl= selected_percent * current_candle_open
            single_candle_sl= single_candle_target-single_percent_sl
            print("single candle sl",single_candle_sl)
            qty=local_quantity
            if single_candle_sl!=ssl:
                ssl=single_candle_sl
                if single_candle_sl > isl_value:            
                    sl_order_id,all_exited= binance2.modify_sl_order_for_sell(name, sl_order_id, single_candle_sl, qty, all_exited)
                    isl_value=single_candle_sl            
               
            if not tslmovement:
                print("if not tslmovement")
                
                trailing_sl_movement_percent= AFTER_SCC_X_PCT_PRICE_MOVE  ##get the values from excel
                trailing_sl_movement= (trailing_sl_movement_percent * current_candle_open)
                trailing_sl_target= single_candle_target+ trailing_sl_movement
                tsl_percent= AFTER_SCC_Y_PCT_TRAILING_MOVE
                single_candle_tsl= (current_candle_open * tsl_percent)
                tslmovement= True
                print('trailing sl target', trailing_sl_target,'single candle tsl',single_candle_tsl)
                
    
        if tslmovement: 
            print("entered into if tslmovement")
            
    
            if o_ltp> trailing_sl_target: 
                print("trailing sl target achieved") 
                
    
                tsl= single_candle_sl+single_candle_tsl
                print('tsl is', tsl)
                
    
                if (((o_ltp-tsl)> (0.03*o_ltp)) and (tsl > isl_value)):
                    print("if ((o_ltp-tsl)> (0.1*o_ltp)):") 
                    print(" shifting x y based tsl")
                   
                    qty=local_quantity
                    sl_order_id,all_exited= binance2.modify_sl_order_for_sell(name, sl_order_id, tsl, qty, all_exited) 
                    isl_value=tsl
                    
                    
    
                else:
                    print(f'single candle sl shifting already happened and {tsl} value is much closer to ltp so tsl not happened')     
                    
                    
         
                trailing_sl_target= trailing_sl_target+trailing_sl_movement
        
        print(" trailing sl revised target",trailing_sl_target)
                      
        return sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value
		
def logic_checking(opt_low, trade, isl_value, sl_order_id, order_exec_price, name, opt_tf, local_quantity, all_exited):
    print("entered into logic_checking")
    
    if (TSL1 == 'Exit half and remaining at bearish candle low' 
        or TSL2 == 'Exit half and remaining at bearish candle low' 
        or TSL1 == 'Exit at market price and half trailing with bearish candle low sl'
        or TSL2 == 'Exit at market price and half trailing with bearish candle low sl'
        or TSL1 == 'Full trailing and half trailing with bearish candle low sl'
        or TSL2 == 'Full trailing and half trailing with bearish candle low sl'):
        bearish_sl = True
        print("entered into Exit half and remaining at bearish candle low")
        
        sl_price = round(float(opt_low), 1)
        price = round(float(opt_low - 0.1), 1)
        
        sl_order_id, all_exited = binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
        
        isl_value = opt_low
        
        return all_exited, sl_order_id, local_quantity, isl_value
		
def full_logic(ce_tradingsymbol, listcheck, user_id, order_id, exchange_id, amount):
    global opt_tf
    
    opt_tf = TIMEFRAME
            
    selected_values = (candle_condition, previous_candle, close_condition,
                      condition, call_candle_condition,
                      call_previous_candle, call_close_condition,
                      call_condition, put_candle_condition,
                      put_previous_candle, put_close_condition,
                      put_condition, ce_tradingsymbol)  ##last strike price from premium pending
    
    call_trade, put_trade, cdf, pdf, listcheck, supertrend_value, CC2, CO2, CH2, CL2, PC2, PO2, PH2, PL2, ce_rsi_df, pe_rsi_df, fut_rsi_df = create_trade_conditions(*selected_values, opt_tf, listcheck)
    print('new', call_trade, put_trade, supertrend_value)
    
    trade_checking = {}
    call_trade_taken = False
    put_trade_taken = False
    print("order type is", ORDER_TYPE)
    name = ce_tradingsymbol

    ###ORDER TYPE LOGICS####
    if call_trade:             
        #trade_ltp,co= get_ltp_and_current_opt_open(name,logFile,opt_tf)
        quantity = get_quantity(name, amount)
        
        qty = quantity
        local_quantity = quantity
        if quantity != 0:
            print("Quantity is", local_quantity)            
            print("checking trade:", name, 'BUY', quantity, type(quantity))
            
            if ORDER_TYPE == 'market':
                order_type = 'market'
                order = binance2.create_market_order('buy', name, quantity)
                price = order["fills"][0]["price"] if order.get("fills") else order["price"]
                order_exec_price = price
                exchange_order_id = order["orderId"]
                exchange_order_status = order["status"]
                exchange_timestamp = order["transactTime"]
                log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Buy", price, quantity, "executed")
                update_order_status(user_id, order_id, "COMPLETE", exchange_order_status)
                
                order_info = get_open_order(order_id)
                status = order_info.get('status')

                if status == 'FILLED':              
                    call_trade_taken = True
                else:
                    print("error in placing order checking conditions again")
                    call_trade = False
                #order_exec_price= orderplacement.get_averageprice(name)
                
                print("order exec price", order_exec_price)           
                target_time = time_calc(opt_tf, entry=False)
                print('target time is', target_time)  ##logic for taking order and getting to know order execution price
                elif ORDER_TYPE=='limit':
                    order_type='limit'
                   

                    print("Type of limit value chosen is", LIMIT_ORDER_VALUE)
                    if LIMIT_ORDER_VALUE=='open':
                        limit_value= int(CO2)
                    elif LIMIT_ORDER_VALUE=='high':
                        limit_value= int(CH2)
                    elif LIMIT_ORDER_VALUE=='low':
                        limit_value= int(CL2)
                    elif LIMIT_ORDER_VALUE=='close':
                        limit_value= int(CC2)
						
                    order = binance2.create_new_order('buy',name, order_type, quantity,limit_value)
					price = order["price"]
					order_exec_price = price
					exchange_order_id = order["orderId"]
					exchange_order_status = order["status"]
					exchange_timestamp = order["transactTime"]
					log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Buy", price, quantity, "executed")
					update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
                    
                    
                    order_info=get_open_order(order_id)
					status=order_info.get('status') 
                    print("The set limit order value is",limit_value)                     
                    target_time=time_calc(opt_tf,entry=True)
                    print("target time is",target_time)
                    nowtimee = datetime.datetime.now()
                    while nowtimee<target_time:
                        order_info=get_open_order(order_id)
						status=order_info.get('status')  
                        if status=='FILLED':
                            call_trade_taken= True
                            target_time=time_calc(opt_tf,entry=False)
                            print("target time for exit at mkt inside limit order",target_time)
                            break
                        elif status=="NEW":
                            call_trade_taken= False
                            put_trade_taken= False
                            print("limit order value is", limit_value)
                            print("limit order status is still open and pending, checking again") 
                        nowtimee = datetime.datetime.now()
                    if (call_trade_taken== False and  put_trade_taken==False):
                        print("Target time is reached going to scan again")
                    if status=="NEW":
                        #cancel_order = kite.cancel_order(order_id = placed_order_id, variety='regular', parent_order_id = None)
                        try:
                            response = binance2.remove_order(name,order_id=order_id)
                            
                            print("Order cancelled successfully", response)
                        except Exception as e:
                            print("Error in cancelling order:", e) 

                        
                elif ORDER_TYPE=='slm':
                    order_type='slm'
                    
                    
                    print("Type of limit value chosen is", LIMIT_ORDER_VALUE)
                    if LIMIT_ORDER_VALUE=='open':
                        limit_value= float(CO2)
                    elif LIMIT_ORDER_VALUE=='high':
                        limit_value= float(CH2)
                    elif LIMIT_ORDER_VALUE=='low':
                        limit_value= float(CL2)
                    elif LIMIT_ORDER_VALUE=='close':
                        limit_value= float(CC2)

                    ltp = get_ltp(name)
                    print("ltp is", ltp)
                    print("limit_value is", limit_value)
                    if limit_value>ltp:
                                       
                    order = binance2.create_slm_order('buy',name, quantity,limit_value)
					exchange_order_id = order["orderId"]
					order_details=binance2.get_order(name, exchange_order_id)
					price = order_details["price"]
					order_exec_price = price
					exchange_order_status = order_details["status"]
					exchange_timestamp = order_details["Time"]
					log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Buy", price, quantity, "executed")
					update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
                    
                    
                    order_info=get_open_order(order_id)
					status=order_info.get('status') 
                        print("status is",status)

                    else:
                        print("ltp is greater than trigger price")
                        status=None

                    if status=='OPEN':
                        print("the status is OPEN,Going to cancel the trade")
                        try:
                            response = binance2.remove_order(name,order_id=order_id)                            
                            print("Order cancelled successfully", response)
                        except Exception as e:
                            print("Error in cancelling order:", e) 

                        status=None

                    if status!="TRIGGER PENDING":                       
                        call_trade_taken=False
                        put_trade_taken=False
                    else:
                        print("slm order has been placed")
                       

                        print("The set limit order value is",limit_value)                     
                        new_logic_slm=False
                        while not new_logic_slm:
                            target_time=time_calc(opt_tf,entry=True)
                            print("target time is",target_time)
                            nowtimee = datetime.datetime.now()
                            while nowtimee<target_time:
									order_info=get_open_order(order_id)
									status=order_info.get('status')
                                if status=='FILLED':
                                    call_trade_taken= True
                                    new_logic_slm=True
                                   
                                    target_time=time_calc(opt_tf,entry=False)
                                    print("target time for exit at mkt inside slm order",target_time)
                                    break
                                elif status=="TRIGGER PENDING":
                                    call_trade_taken= False
                                    put_trade_taken= False
                                    print("slm order value is", limit_value)
                                    print("slm order status is still trigger pending, checking again") 
                                nowtimee = datetime.datetime.now()
                            if (call_trade_taken== False and  put_trade_taken==False):
                                print("Target time is reached going to scan again")

                            if status=="TRIGGER PENDING":
                                #cancel_order = kite.cancel_order(order_id = placed_order_id, variety='regular', parent_order_id = None)
                                # time.sleep(3)
                                # rsi_length= individual_values[f'rsi_{i}']                               
                                # CC1,CC2,CO2,CH2,CL2,CRSI,cdf,prev_crsi,callp1_time= get_data(name,opt_tf,4,'NFO',logFile,rsi_length) 
                                # if CO2 > CC2:
                                #     print("previous candle time is",callp1_time)
                                #     print("previous candle open is",CO2)
                                #     print("previous candle close is",CC2)
                                #     print("the last candle is bearish so canceling the existing slm")                                             
                                try:
                                    response = binance2.remove_order(name,order_id=order_id)
                                    
                                    print("Order cancelled successfully", response)
                                except Exception as e:
                                    print("Error in cancelling order:", e)                                    
                                    
                                # else:
                                #     print("the candle is bullish so not cancelling slm order")

                            new_logic_slm=True

              else:
                print('The Quantity is not Valid. So no trade taken!!');time.sleep(3)
                call_trade = False
            elif put_trade:
              #trade_ltp,co= get_ltp_and_current_opt_open(name,opt_tf)
              #quantity= get_quantity(name,trade_ltp)
              quantity= get_quantity(name,amount)
              qty=quantity
              local_quantity=quantity
              if quantity != 0:
                if ORDER_TYPE=='market':
                    order_type= 'market'
                    
                    limit_value=0
                    print("Quantity is",local_quantity)
                    order = binance2.create_market_order('buy',name, quantity)
					price = order["fills"][0]["price"] if order.get("fills") else order["price"]
					order_exec_price = price
					exchange_order_id = order["orderId"]
					exchange_order_status = order["status"]
					exchange_timestamp = order["transactTime"]
					
					log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Buy", price, quantity, "executed")
					
					update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
                    
                    
                    order_info=get_open_order(order_id)
					status=order_info.get('status')
                    if status=='FILLED':              
                        put_trade_taken= True
                    else:
                        print("error in placing order checking conditions again")
                        put_trade=False
                    
                    target_time= time_calc(opt_tf,entry=False)
                    print("target time is", target_time)


                elif ORDER_TYPE=='limit':
                    order_type='limit'
                    

                    print("Type of limit value chosen is", LIMIT_ORDER_VALUE)
                    if LIMIT_ORDER_VALUE=='open':
                        limit_value= int(PO2)
                    elif LIMIT_ORDER_VALUE=='high':
                        limit_value= int(PH2)
                    elif LIMIT_ORDER_VALUE=='low':
                        limit_value= int(PL2)
                    elif LIMIT_ORDER_VALUE=='close':
                        limit_value= int(PC2)

                    print("The set limit order value is",limit_value)                     

                    order = binance2.create_new_order('buy',name, order_type, quantity,limit_value)
					price = order["price"]
					order_exec_price = price
					exchange_order_id = order["orderId"]
					exchange_order_status = order["status"]
										
					log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Buy", price, quantity, "executed")
					
					update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
                                        
                    order_info=get_open_order(order_id)
					status=order_info.get('status') 
                                            
                    target_time=time_calc(opt_tf,entry=True)
                    print("target time is",target_time)
                    nowtimee = datetime.datetime.now()
                    while nowtimee<target_time:
							order_info=get_open_order(order_id)
							status=order_info.get('status')  
                        if status=='FILLED':
                            put_trade_taken= True
                            target_time=time_calc(opt_tf,entry=False)
                            print("target time for exit at mkt inside limit order",target_time)
                            break
                        elif status=="NEW":
                            call_trade_taken= False
                            put_trade_taken= False
                            print("limit order value is", limit_value)
                            print("limit order status is still open and pending, checking again") 
                        nowtimee = datetime.datetime.now()
                    if status=="NEW":
                        #cancel_order = kite.cancel_order(order_id = placed_order_id, variety='regular', parent_order_id = None)
                        try:
                            response = binance2.remove_order(name,order_id=order_id)
                            print("Order cancelled successfully", response)
                        except Exception as e:
                            print("Error in cancelling order:", e)

                elif ORDER_TYPE=='slm':
                    order_type='slm'
                   
                    
                  
                    print("Type of limit value chosen is", LIMIT_ORDER_VALUE)
                    if LIMIT_ORDER_VALUE=='open':
                        limit_value= float(PO2)
                    elif LIMIT_ORDER_VALUE=='high':
                        limit_value= float(PH2)
                    elif LIMIT_ORDER_VALUE=='low':
                        limit_value= float(PL2)
                    elif LIMIT_ORDER_VALUE=='close':
                        limit_value= float(PC2)

                    ltp = get_ltp(name)
                    print("ltp is", ltp)
                    print("limit_value is", limit_value)
                    if limit_value>ltp:
                        print("The set limit order value is",limit_value) 
                        order = binance2.create_new_order('buy',name, order_type, quantity,limit_value,limit_value)
						exchange_order_id = order["orderId"]
						order_details=binance2.get_order(name, exchange_order_id)
						price = order_details["price"]
						order_exec_price = price
						exchange_order_status = order_details["status"]
						exchange_timestamp = order_details["Time"]
						log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Buy", price, quantity, "executed")
						
						update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
						order_info=get_open_order(order_id)
						status=order_info.get('status') 
                        print("status is",status)
                    else:
                        print("ltp is greater than trigger price")
                        status=None


                    if status=='OPEN':
                        print("the status is OPEN,Going to cancel the trade")
                        try:
                            response = binance2.remove_order(name,order_id=order_id)                            
                            print("Order cancelled successfully", response)
                        except Exception as e:
                            print("Error in cancelling order:", e) 

                        status=None

                    if status!="TRIGGER PENDING":                       
                        call_trade_taken=False
                        put_trade_taken=False
                    
                    else:
                        order_info=get_open_order(order_id)
						status=order_info.get('status') 
                        new_logic_slm=False
                        while not new_logic_slm:                        
                            target_time=time_calc(opt_tf,entry=True)
                            print("target time is",target_time)
                            nowtimee = datetime.datetime.now()
                            while nowtimee<target_time:
                                order_info=get_open_order(order_id)
								status=order_info.get('status') 
                                if status=='FILLED':
                                    put_trade_taken= True
                                    new_logic_slm=True
                                    
                                    target_time=time_calc(opt_tf,entry=False)
                                    print("target time for exit at mkt inside limit order",target_time)
                                    break
                                elif status=="TRIGGER PENDING":
                                    call_trade_taken= False
                                    put_trade_taken= False
                                    print("slm order value is", limit_value)
                                    print("slm order status is still trigger pending, checking again") 
                                nowtimee = datetime.datetime.now()
                            
                            if status=="TRIGGER PENDING":
                                #cancel_order = kite.cancel_order(order_id = placed_order_id, variety='regular', parent_order_id = None)
                                # rsi_length= individual_values[f'rsi_{i}']                               
                                # CC1,CC2,CO2,CH2,CL2,CRSI,cdf,prev_crsi,callp1_time= get_data(name,opt_tf,4,'NFO',logFile,rsi_length) 
                                # if CO2 > CC2:
                                #     print("previous candle time is",callp1_time)
                                #     print("previous candle open is",CO2)
                                #     print("previous candle close is",CC2)
                                #     print("the last candle is bearish so canceling the existing slm")
                                try:
                                    response = binance2.remove_order(name,order_id=order_id)
                                    print("Order cancelled successfully", response)
                                except Exception as e:
                                    print("Error in cancelling order:", e)
                            new_logic_slm=True
                               


              else:
                print('The Quantity is not Valid. So no trade taken!!');time.sleep(3)
                put_trade = False
            isl=False
            all_exited=False
            




            ###### SL LOGICS #####
            while ((call_trade_taken or put_trade_taken) and (not all_exited)): #make the loop run only if there is open quantities on that specific order
              
                if not isl:
                                 
                    if (ISL== 'Call Option Candle Lowest Between P-2 and P-1 candle low' and not all_exited) :                       
                        
                        if call_trade_taken:
                            sl=get_isl(cdf,order_exec_price) 
                            # sl=10                                             
                                                  
                        if put_trade_taken:
                            sl=get_isl(pdf,order_exec_price)                    
                        
    
                        sl_price= round(float(sl),1)
                        isl_value=sl_price
                        print("isl is",sl_price)
                       
    
                        price= round(float(sl-0.1),1)
                        ltp= get_ltp(name)
                        if sl_price>ltp:
                            print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                                                       
                            limit_value=0                               
                            
                            
                            order_info=get_open_order(order_id)
							status=order_info.get('status')
                            if status=="FILLED":
                                all_exited=True
                                
                           
                        else:
                            
                            
                                    
                             
							order = binance2.create_slm_order('sell',name, quantity,limit_value)
							sl_order_id = order["orderId"]
							order_details=binance2.get_order(name, sl_order_id)
							price = order_details["price"]
							order_exec_price = price
							exchange_order_status = order_details["status"]
							exchange_timestamp = order_details["Time"]
							log_trade(user_id, order_id, exchange_id, sl_order_id, name, order_type, "Simple Sell", price, quantity, "executed")
							update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
							
							
							order_info=get_open_order(order_id)
							status=order_info.get('status') 


                            #sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(all_exited,name,quantity,kite.ORDER_TYPE_SL,kite.PRODUCT_MIS,price,kite.TRANSACTION_TYPE_SELL,kite.EXCHANGE_NFO,kite.VARIETY_REGULAR,sl_price,kite.VALIDITY_DAY,logFile,kite,Order_Identifier)                    
                            

                            print("sl order id", sl_order_id)
                            print("placed isl")
                            
                        c=0;o=0;rsi=0;opt_open=0;opt_close=0;orsi=0                    
                        isl=True
                        
    
                    elif (ISL== 'Call Option Last Bearish Candle Low' and not all_exited):
                        
                        if call_trade_taken:
                            sl=get_last_bearish_candle(cdf,order_exec_price)
                                               
                        if put_trade_taken:
                            sl=get_last_bearish_candle(pdf,order_exec_price)
                        
                        sl_price= round(float(sl),1)
                        isl_value= sl_price
                        print("isl is",sl_price)
    
                        price= round(float(sl-0.1),1) 
                        ltp= get_ltp(name)
                        if sl_price>ltp:
                            print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                            
                      
                            limit_value=0                              
                            order = binance2.create_new_order('sell',name, order_type, quantity,limit_value)
							price = order["price"]
							order_exec_price = price
							exchange_order_id = order["orderId"]
							exchange_order_status = order["status"]
							exchange_timestamp = order["transactTime"]
							log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Sell", price, quantity, "executed")
							update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)


							order_info=get_open_order(order_id)
							status=order_info.get('status')
                            if status=="FILLED":
                                all_exited=True
                                
                        else:

                            
                            order = binance2.create_slm_order('sell',name, quantity,limit_value)
							sl_order_id = order["orderId"]
							order_details=binance2.get_order(name, sl_order_id)
							price = order_details["price"]
							order_exec_price = price
							exchange_order_status = order_details["status"]
							exchange_timestamp = order_details["Time"]
							log_trade(user_id, order_id, exchange_id, sl_order_id, name, order_type, "Simple Sell", price, quantity, "executed")
							update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
							
							
							order_info=get_open_order(order_id)
							status=order_info.get('status')        
                     
                            

                            
                            print("sl order id", sl_order_id)
                            print("placed isl")
                            
                        
                        c=0;o=0;rsi=0;opt_open=0;opt_close=0;orsi=0                   
                                        
                        isl=True

                    elif (ISL== 'Not applicable' and not all_exited):
						Isl_percentage = float(Isl_percentage.strip('%')) / 100
                        sl=(order_exec_price - (Isl_percentage *order_exec_price))
                        sl_price= round(float(sl),1)
                        isl_value= sl_price
                        print("isl is",sl_price)
                        price= round(float(sl-0.1),1)
                       
                        ltp= get_ltp(name)
                        if sl_price>ltp:
                            print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                            
                            
                            limit_value=0                                
                            order = binance2.create_market_order('sell',name, qty)
							price = order["fills"][0]["price"] if order.get("fills") else order["price"]
							order_exec_price = price
							exchange_order_id = order["orderId"]
							exchange_order_status = order["status"]
							exchange_timestamp = order["transactTime"]
							log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Sell", price, qty, "executed")
							update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
                    
                    
							order_info=get_open_order(order_id)
							status=order_info.get('status')
                            if status=="FILLED":
                                print("exited")
                        else:                                    
                            
                           
                            
                            sl_order_id,all_exited
							order = binance2.create_slm_order('sell',name, quantity,sl_price)
							sl_order_id = order["orderId"]
							order_details=binance2.get_order(name, sl_order_id)
							price = order_details["price"]
							order_exec_price = price
							exchange_order_status = order_details["status"]
							exchange_timestamp = order_details["Time"]
							log_trade(user_id, order_id, exchange_id, sl_order_id, name, order_type, "Simple Sell", price, quantity, "executed")
							update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
                            if all_exited:
                                print("exited the trade")

                            
                            print("sl oredr id", sl_order_id)
                            print("placed isl")
                            
                        
                        c=0;o=0;rsi=0;opt_open=0;opt_close=0;orsi=0                   
                        
                
                        isl=True
    
                    elif (ISL== 'Bearish candle below LL' and not all_exited):
                        if call_trade_taken:
                            sl=get_last_bearish_candle_with_rsi(ce_rsi_df,order_exec_price,RSI_LOWER)
                                               
                        if put_trade_taken:
                            sl=get_last_bearish_candle_with_rsi(pe_rsi_df,order_exec_price,RSI_LOWER)
                        
                        sl_price= round(float(sl),1)
                        isl_value= sl_price
                        print("isl is",sl_price)
    
                        price= round(float(sl-0.1),1) 
                        ltp= get_ltp(name)
                        if sl_price>ltp:
                            print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                            
                            
                            limit_value=0                              
                            order = binance2.create_market_order('buy',name, quantity)
							price = order["fills"][0]["price"] if order.get("fills") else order["price"]
							order_exec_price = price
							exchange_order_id = order["orderId"]
							exchange_order_status = order["status"]
							exchange_timestamp = order["transactTime"]
							log_trade(user_id, order_id, exchange_id, exchange_order_id, name, order_type, "Simple Buy", price, quantity, "executed")
							update_order_status(user_id, order_id, "COMPLETE",exchange_order_status)
							
							
							order_info=get_open_order(order_id)
							status=order_info.get('status')
                            if status=="FILLED":
                                all_exited=True
                                
                        else:

                            
                            order_type="slm"
                             
							order = binance2.create_slm_order('sell',name, quantity,limit_value)
							sl_order_id = order["orderId"]
							order_details=binance2.get_order(name, sl_order_id)
							price = order_details["price"]
							order_exec_price = price
							exchange_order_status = order_details["status"]
							exchange_timestamp = order_details["Time"]
							log_trade(user_id, order_id, exchange_id, sl_order_id, name, order_type, "Simple Sell", price, quantity, "executed")
                            

                            
                            print("sl order id", sl_order_id)
                            print("placed isl")
                            
                        
                        c=0;o=0;rsi=0;opt_open=0;opt_close=0;orsi=0                   
                        
                
                        isl=True

                else:
                    tslmovement= False;trailing_sl_movement=0;trailing_sl_target=0;single_candle_tsl=0;single_candle_sl=0;ssl=0
                    ts1_condition_satisfied= None
                    ts2_condition_satisfied= None
                    entered_condition = None
                    exit_at_mkt=False
                    bull_condt=None
                    bullish_rsi=None
                    first_target=True
                    second_target=False
                    target_order_placed=False
                    target_order_id=None
                    p1=[]
                    while not all_exited:
    
                        #status=orderplacement.get_orderstatus(sl_order_id,logFile)
                        
						order_details=binance2.get_order(name, sl_order_id)
						status = order_details["status"]
                        print("status of SL order", status)
                        
    
                        if status=='FILLED':
                            all_exited=True
                            if target_order_id !=None:
                                try:
                                    response = response = binance2.remove_order(name,order_id=target_order_id)                            
                                    print("Order cancelled successfully", response)
                                except Exception as e:
                                    print("Error in cancelling order:", e)


                        elif status=='OPEN':
                            print("status got changed to open so exiting all")
                           # oid= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
                            all_exited= True
                            local_quantity=0


                        current_time= datetime.datetime.now().time()
                        exit_time= EXIT_TIME                       
                        if current_time>=exit_time:
                            #oid= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
                            time.sleep(5)
                            status= binance2.get_order(name,oid)
                            #order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_xec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                            if status=='FILLED':
                                all_exited=True
                                local_quantity=0
                                if target_order_id !=None:
                                    try:
                                        response = binance2.remove_order(name,order_id=target_order_id)                            
                                        print("Order cancelled successfully", response)
                                    except Exception as e:
                                        print("Error in cancelling order:", e)
                                
                                print("Time is greater than exit time exiting all")
                                                                                                                                           
                        else:
                            
                            now= datetime.datetime.now()
                            if 4 < now.second < 50 and not all_exited:
                                rsi_length= RSI_LENGTH
                                new_close, new_open, new_rsi, prev_rsi,new_low,last_candle_time = get_close_and_rsi(name,bnf_tf,4,rsi_length )
                                new_opt_close, new_opt_open, new_orsi, prev_orsi,new_opt_low,last_candle_time= get_close_and_rsi(name,opt_tf,4,rsi_length)
                                Cll1,Cll2,clO2,clH2,clL2,clRSI,nadf,cprev_rsi,ccandle_p1_time,rsi_rsi= get_data(name,bnf_tf,rsi_length)
                                ts1_condition_satisfied=False ; ts2_condition_satisfied=False
    
                                                        
                                
                                
                                if (call_trade_taken and (prev_orsi>RSI_UPPER)):
                                     ts1_condition_satisfied= True
                                     print('CC','prev_rsi:',prev_rsi, 'prev_orsi:',prev_orsi,'UL:',RSI_UPPER)
                                
                                if (call_trade_taken  and (prev_orsi>RSI_LOWER)):
                                     ts2_condition_satisfied= True
                                     print('CC','prev_rsi:',prev_rsi, 'prev_orsi:',prev_orsi,'UL:',RSI_UPPER)
                                
                                if (put_trade_taken  and (prev_orsi>RSI_UPPER)):
                                     ts1_condition_satisfied= True
                                     print('PC','prev_rsi:',prev_rsi, 'prev_orsi:',prev_orsi,'UL:',RSI_UPPER, 'LL:', RSI_LOWER)
    
                                if (put_trade_taken and (prev_orsi>RSI_LOWER)):
                                     ts2_condition_satisfied= True
                                     print('PC','prev_rsi:',prev_rsi, 'prev_orsi:',prev_orsi,'UL:',RSI_UPPER, 'LL:', RSI_LOWER)
    
                                print("ts1 condition:",ts1_condition_satisfied,"ts2 condition:",ts2_condition_satisfied)
                               
                                                                                    
                                if ((not p1 or p1[0] != last_candle_time) and not all_exited):
                                    p1.clear()
                                    p1.append(last_candle_time)
                                    bull_condt=True
                                    print("p1 candle time is",p1)
                                    c, o, rsi,fut_low,prevrsi = new_close, new_open, new_rsi,new_low,prev_rsi   
                                    opt_close, opt_open, orsi,opt_low = new_opt_close, new_opt_open, new_orsi, new_opt_low
                                    print('values got changed new values are',c,o,rsi,opt_close,opt_open,orsi,opt_low,fut_low)
                                    print_str = f"Current Time: {datetime.datetime.now()}  \n"
                                    print_str += f"1. {name} P1 candle close: {opt_close} | 2. {name} P1 candle open: {opt_open} | 3. {name} P1 candle RSI: {orsi} | 4. {name} P1 candle low: {opt_low} | 5. P1 {name} candle time: {last_candle_time} | 6. {name} P2 RSI:{prev_orsi} \n"
                                    print_str += f"7. {index_future} P1 candle close: {c} | 8. {index_future} P1 candle open: {o} | 9. {index_future} P1 candle RSI: {rsi} | 10. {index_future} P1 candle low: {fut_low} | 11. P1 {index_future} candle time: {last_candle_time} | 12. {index_future} P2 RSI:{prevrsi} \n"
                                    print('\nCondition checking!:',print_str,'UL-->:',RSI_UPPER,'LL-->:',RSI_LOWER)                                   
                                    print('last candle time', last_candle_time)
                                    st=get_supertrend(nadf,logFile,i)
                                    print("Future super trend value is",st)
                                                                 
                                    entered_condition = False
                                    bullish_rsi=True

                        
                                print_str = f"Current Time: {datetime.datetime.now()}  \n"
                                print_str += f"1. {name} P1 candle close: {opt_close} | 2. {name} P1 candle open: {opt_open} | 3. {name} P1 candle RSI: {orsi} | 4. {name} P1 candle low: {opt_low} | 5. P1 {name} candle time: {last_candle_time} | 6. {name} P2 RSI:{prev_orsi} \n"
                                print_str += f"7. {index_future} P1 candle close: {c} | 8. {index_future} P1 candle open: {o} | 9. {index_future} P1 candle RSI: {rsi} | 10. {index_future} P1 candle low: {fut_low} | 11. P1 {index_future} candle time: {last_candle_time} | 12. {index_future} P2 RSI:{prev_rsi} \n"
                                print('\nCondition checking!:',print_str,'UL-->:',RSI_UPPER,'LL-->:',RSI_LOWER)
                                


                                #### 1A/1B AND TARGET LOGICS #####
                                if ((TSL1 == 'Exit at market price' or TSL2 == 'Exit at market price')  and not all_exited):                       
                                    now = datetime.datetime.now()
                                    if now> target_time and not all_exited:
                                        if orsi > BULLISH_RSI_LIMIT and bullish_rsi and BULLISH_RSI_ENABLER=="enable":
                                            if opt_low > order_exec_price:
                                                bullish_rsi=False
                                                print(f"p1 rsi: {orsi} is greater than the given rsi value:{BULLISH_RSI_LIMIT}")
                                                sl_price= round(float(opt_low),1)
                                                price= round(float(opt_low-0.1),1) 
                                               
												binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
                                                isl_value=opt_low
                                                print("Shifting rsi based tsl value")
                                                print("New tsl is",isl_value)
                                               
                                            else:
                                                print("option low is not greater than order executed price so not shifting rsi based sl")
                                    
                                        elif opt_close < opt_open:                                    
                                            all_exited,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,local_quantity,ssl=checking_condition(trade,ssl,opt_low,order_exec_price,name,opt_tf,sl_order_id,all_exited,local_quantity,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                        elif opt_close > opt_open and BULLISH_RSI_ENABLER=="enable":
                                            print("P1 candle is bullish so checking the p2 and p3 candles")
                                            sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,i,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,rsi_length)
                                            
                                    else:                        
                                        if not all_exited:
                                            sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                            ltpp,high=get_ltp_and_current_opt_high(name,opt_tf)
                                            o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
                                            all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity)
                        
                                #### 2A/2B AND TARGET LOGICS #####
                                elif ((TSL1== 'Full trailing with bearish candle low sl') or (TSL2== 'Full trailing with bearish candle low sl'))and not all_exited:
                                    
                                    if orsi > BULLISH_RSI_LIMIT and bullish_rsi and BULLISH_RSI_ENABLER=="enable":
                                        if opt_low > order_exec_price:
                                            
                                            bullish_rsi=False
                                            print(f"p1 rsi: {orsi} is greater than the given rsi value:{BULLISH_RSI_LIMIT}")
                                            sl_price= round(float(opt_low),1)
                                            price= round(float(opt_low-0.1),1) 
                                            sl_order_id,all_exited= binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
                                            
                                            isl_value=opt_low
                                            print("Shifting rsi based tsl value")
                                            print("New tsl is",isl_value)
                                            
                                        else:
                                            print("option low is not greater than order executed price so not shifting rsi based sl")
                                    
                                    
                                    elif opt_close < opt_open:
                                        print("p1 candle is bearish")
                                        

                                        sl_order_id,isl_value,all_exited= full_trailing_checking(name,trade,opt_low,isl_value,sl_order_id,all_exited,local_quantity)
                                    
                                    elif opt_close > opt_open and BULLISH_RSI_ENABLER=="enable":
                                        print("P1 candle is bullish so checking the p2 and p3 candles")
                                        sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,rsi_length)
                                                                        
                                    if not all_exited:
                                        sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                        o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
                                        ltpp,high=get_ltp_and_current_opt_high(name,opt_tf)
                                        
                                        all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity)
        
                                #### 3A/3B AND TARGET LOGICS #####
                                elif ((TSL1== 'Exit half and remaining at bearish candle low') or (TSL2== 'Exit half and remaining at bearish candle low') or (TSL2== 'Exit half and remaining at bearish candle low above sl price'))  and not all_exited:
                                    
                                    if orsi > BULLISH_RSI_LIMIT and bullish_rsi and BULLISH_RSI_ENABLER=="enable":
                                        if opt_low > order_exec_price:
                                                                            
                                            bullish_rsi=False
                                            print(f"p1 rsi: {orsi} is greater than the given rsi value:{BULLISH_RSI_LIMIT}")
                                            sl_price= round(float(opt_low),1)
                                            price= round(float(opt_low-0.1),1) 
                                            sl_order_id,all_exited= binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
                                            

                                            isl_value=opt_low
                                            print("Shifting rsi based tsl value")
                                            print("New tsl is",isl_value)
                                            
                                        else:
                                            print("option low is not greater than order executed price so not shifting rsi based sl")
                                    
                                    
                                    elif opt_close < opt_open:
                                        
                                        if opt_low > order_exec_price:
                                            
                                            
                                            print("opt close is greater than order executed price and p1 candle is bearish")
                                            print("opt close is",opt_close,"order executed price is",order_exec_price)
                                            
                                            if TSL1== 'Exit half and remaining at bearish candle low':
                                                print(" 3A is chosen")

                                                all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                            elif SUPERTREND_TRAILING=="enable" and TSL2== 'Exit half and remaining at bearish candle low':
                                                print("super trend based trailing is enabled and 3B is chosen")
                                                
                                                if trade=="call" and st=="down":
                                                    print("call trade is taken and super trend is red so applying 3B")

                                                    all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                                else:
                                                    print("call trade is taken and super trend is still green so not applying 3A/3B")
                                                if trade=="put" and st=="up":
                                                    print("put trade is taken and super trend is green so applying 3A")

                                                    all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                                else:
                                                    print("put trade is taken and super trend is still red so not applying 3A/3B")

                                            elif SUPERTREND_TRAILING=="disable" and TSL2== 'Exit half and remaining at bearish candle low':
                                                print(" 3B is chosen and st is disable")
                                                all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)

                                        elif  opt_low > isl_value and TSL2== 'Exit half and remaining at bearish candle low above sl price':
                                            print("option low is",opt_low)
                                            print("isl_value is",isl_value)
                                            print("Option low is greater than isl so going to apply 3b")
                                            if trade=="call" and st=="down":
                                                    print("call trade is taken and super trend is red so applying 3B")

                                                    all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                            else:
                                                print("call trade is taken and super trend is still green so not applying 3A/3B")
                                            if trade=="put" and st=="up":
                                                print("put trade is taken and super trend is green so applying 3A")

                                                all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                            else:
                                                print("put trade is taken and super trend is still red so not applying 3A/3B")

                                            

                                            
                                                                                        
                                        else:
                                            print(" option close is not greater than order executed price so not executing tsl1")

                                    elif opt_close > opt_open and BULLISH_RSI_ENABLER=="enable":
                                        print("P1 candle is bullish so checking the p2 and p3 candles")
                                        sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,rsi_length)
                                    

                                    if not all_exited:
                                        sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                        o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
                                        ltpp,high=get_ltp_and_current_opt_high(name,opt_tf)

                                        all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity)
        

                                #### 4A/4B AND TARGET LOGICS #####
                                elif ((TSL1== 'Exit at market price and half trailing with bearish candle low sl') or (TSL2== 'Exit at market price and half trailing with bearish candle low sl')) and not all_exited:
                                    now = datetime.datetime.now()
                                    if now> target_time and not all_exited:
                                        if orsi > BULLISH_RSI_LIMIT and bullish_rsi and BULLISH_RSI_ENABLER=="enable":
                                            
                                            if opt_low > order_exec_price:

                                                bullish_rsi=False
                                                print(f"p1 rsi: {orsi} is greater than the given rsi value:{BULLISH_RSI_LIMIT}")
                                                sl_price= round(float(opt_low),1)
                                                price= round(float(opt_low-0.1),1) 
                                                sl_order_id,all_exited= binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
                                                
                                                isl_value=opt_low
                                                print("Shifting rsi based tsl value")
                                                print("New tsl is",isl_value)
                                                
                                            else:
                                                print(" option close is not greater than order executed price so not executing tsl1")
                                        
                                        elif opt_close < opt_open:
                                            if opt_low < order_exec_price and not all_exited:
                                                exit_all= binance2.modify_mkt_order(sl_order_id,local_quantity,name)
                                                #status=orderplacement.get_orderstatus(exit_all,logFile)
												exit_all = exit_all["orderId"]
                                                status = binance2.get_order(name,exit_all)
												status = status["status"]
                                                if status=='FILLED':
                                                    all_exited= True
                                                    
                                                    print("exited all")
                                                                            
                                                    local_quantity=0
                                            else:
                                                if not entered_condition:
                                                    
                                                    all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                                    

                                        elif opt_close > opt_open and BULLISH_RSI_ENABLER=="enable":
                                            print("P1 candle is bullish so checking the p2 and p3 candles")
                                            sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,rsi_length)
                                    
                                        
                                        if not all_exited:
                                                sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                                o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
                                                ltpp,high=get_ltp_and_current_opt_high(name,opt_tf)
                                                all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity)
                                    else:
                                        if not all_exited:
                                                sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                                o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
                                                ltpp,high=get_ltp_and_current_opt_high(name,opt_tf)
                                                all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity)
                
                                
                                
                                #### 5A/5B AND TARGET LOGICS #####
                                elif ((TSL1== 'Full trailing and half trailing with bearish candle low sl') or (TSL2== 'Full trailing and half trailing with bearish candle low sl')) and not all_exited:
                                    if orsi > BULLISH_RSI_LIMIT and bullish_rsi and BULLISH_RSI_ENABLER=="enable":
                                            
                                        if opt_low > order_exec_price:
                                            
                                            bullish_rsi=False
                                            print(f"p1 rsi: {orsi} is greater than the given rsi value:{BULLISH_RSI_LIMIT}")
                                            sl_price= round(float(opt_low),1)
                                            price= round(float(opt_low-0.1),1) 
                                            sl_order_id,all_exited= binance2.modify_stop_loss(name, sl_order_id, sl_price, price, local_quantity,all_exited)
                                            

                                            isl_value=opt_low
                                            print("Shifting rsi based tsl value")
                                            print("New tsl is",isl_value)
                                            
                                        else:
                                                print(" option close is not greater than order executed price so not executing tsl1")
                                    
                                    elif opt_close < opt_open:
                                        if opt_low < order_exec_price:
                                            sl_order_id,isl_value,all_exited= full_trailing_checking(name,trade,opt_low,isl_value,sl_order_id,all_exited,local_quantity)
                                            
                                        else:
                                            all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl_with_extra_condition(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                    
                                    elif opt_close > opt_open and BULLISH_RSI_ENABLER=="enable":
                                        print("P1 candle is bullish so checking the p2 and p3 candles")
                                        sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,rsi_length)
                                    
                                    
                                    if not all_exited:
                                            sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                            o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
                                            ltpp,high=get_ltp_and_current_opt_high(name,opt_tf)
                                            all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity)
            
                            else:
                                if not all_exited:
                                    sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                    o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
                                    ltpp,high=get_ltp_and_current_opt_high(name,opt_tf)
                                    all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity)

                    t_time= time_calc(i,opt_tf,True)
                    now = datetime.datetime.now()
                    while now<t_time:
                        
                        now = datetime.datetime.now()

                        
                    print("Going to take trade")


            return listcheck
            
# Simple Buy Function
def simple_buy(user_id, order_id, exchange_id, symbol, amount):
    df = fetch_ohlcv(symbol)
    ha_df = convert_to_heikin_ashi(df)
    ha_df["rsi"] = calculate_rsi(ha_df)

    last_candle = ha_df.iloc[-2]
    current_candle = ha_df.iloc[-1]
    
    green_candle = last_candle["ha_close"] > last_candle["open"]
    red_candle = last_candle["open"] > last_candle["ha_close"]
    
    min_notional = get_min_notional(symbol)  # Get min trade value in USDT
    min_amount = get_min_trade_amount(symbol)  # Get min amount for the pair
    step_size = get_step_size(symbol)
    
    last_price = get_ltp(symbol)
    min_notional = Decimal(str(min_notional))  # Convert min_notional to Decimal
    step_size = Decimal(str(step_size))
    amount = Decimal(amount)
    last_price = Decimal(str(last_price))  # Convert last_price to Decimal
    order_value = amount * last_price  # Multiply correctly

    # Adjust if below min_notional
    if order_value < min_notional:
        adjusted_amount = min_notional / last_price
        adjusted_amount = math.floor(adjusted_amount / step_size) * step_size  # Round to step size
       # print(f"Adjusting amount from {amount} to {adjusted_amount} to meet min notional.")
        amount = adjusted_amount

    if amount < min_amount:
        print(f"Final amount {amount} is still below min trade size. Order skipped.")
        return
		
		#entry conditions

    if green_candle and current_candle["ha_close"] > last_candle["ha_close"] and ha_df["rsi"].iloc[-1] < RSI_LOWER :
        exchange_order_id=0
        try:
            order = binance2.create_market_order('buy',symbol, amount)
            #order = binance.create_market_buy_order(symbol, amount)
            price = order["fills"][0]["price"] if order.get("fills") else order["price"]
            exchange_order_id= order["orderId"]
            order_type= "market"
			
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, order_type, "Simple Buy", price, amount, "executed")
            update_order_status(user_id, order_id, "Complete")
            print(f"BUY ORDER EXECUTED for User {user_id}: {order}")

        except Exception as e:
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, "Simple Buy", 0, amount, "failed")
            print(f"BUY ORDER FAILED for User {user_id}: {e}")
    else:
        print(f"No Buy Signal for User {user_id} for {symbol}")

# Simple Sell Trade Function using Heikin-Ashi
def simple_sell(user_id, order_id, exchange_id, symbol, amount):
    df = fetch_ohlcv(symbol)
    ha_df = convert_to_heikin_ashi(df)
    ha_df["rsi"] = calculate_rsi(ha_df)

    last_candle = ha_df.iloc[-2]
    current_candle = ha_df.iloc[-1]
    
    red_candle = last_candle["open"] > last_candle["ha_close"]  # Fixed indentation

    min_notional = get_min_notional(symbol)  # Get min trade value in USDT
    min_amount = get_min_trade_amount(symbol)  # Get min amount for the pair
    step_size = get_step_size(symbol)
    # Fetch latest price to calculate notional value
    
    last_price = get_ltp(symbol)
    min_notional = Decimal(str(min_notional))  # Convert min_notional to Decimal
    step_size = Decimal(str(step_size))
    amount = Decimal(amount)
    last_price = Decimal(str(last_price))  # Convert last_price to Decimal
    order_value = amount * last_price  # Multiply correctly

    # Adjust if below min_notional
    if order_value < min_notional:
        adjusted_amount = min_notional / last_price
        adjusted_amount = math.floor(adjusted_amount / step_size) * step_size  # Round to step size
       # print(f"Adjusting amount from {amount} to {adjusted_amount} to meet min notional.")
        amount = adjusted_amount

    if amount < min_amount:
        print(f"Final amount {amount} is still below min trade size. Order skipped.")
        return

    if red_candle and current_candle["ha_close"] < last_candle["ha_close"] and ha_df["rsi"].iloc[-1] > RSI_UPPER:
        exchange_order_id=0
        try:
            #order = binance.create_market_sell_order(symbol, amount)
            order = binance2.create_market_order('sell',symbol, amount)
            price = order["fills"][0]["price"] if order.get("fills") else order["price"]
            exchange_order_id = order["orderId"]
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, "Simple Sell", price, amount, "executed")
            update_order_status(user_id, order_id, "Complete")
           # print(f"exchange_order_id: {exchange_order_id}")
            print(f"SELL ORDER EXECUTED for User {user_id}: {order}")

        except Exception as e:
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, "Simple Sell", 0, amount, "failed")
            print(f"SELL ORDER FAILED for User {user_id}: {e}")
    else:
        print(f"No Sell Signal for User {user_id} for {symbol}")


def process_pending_orders():
    while True:
        conn = db_connect()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM trades WHERE status = 'Pending'")
        orders = cursor.fetchall()
        conn.close()

        for order in orders:
            user_id = order["user_id"]
            order_id = order["id"]
            exchange_id = order["exchange_id"]
            symbol = order["currency"]
            order_type = order["type"]
            amount = order["qty"]

            if order_type == "Simple Buy":
                simple_buy(user_id, order_id, exchange_id, symbol, amount)
            elif order_type == "Simple Sell":
                simple_sell(user_id, order_id, exchange_id, symbol, amount)

        time.sleep(3)
# Example Usage
if __name__ == "__main__":
    #process_pending_orders()
	
  #df= fetch_ohlcv("BNBUSDT")
  #get_isl(df,2015)
  #get_last_bearish_candle(df,2015.50)
  #RSI,prev_rsi,rsi_df = get_rsi_df(df,RSI_LENGTH)
  symbol="BNBUSDT"
  #CC1,CC2,CO2,CH2,CL2,CRSI,cdf,prev_crsi,callp1_time,ce_rsi_df = get_data(symbol, TIMEFRAME, RSI_LENGTH)
 
  #dd=calculate_supertrend(cdf, int(length), int(factor))
 
  #latest_trend = dd["trend"].iloc[-1]
  #result = data_analysis(symbol,TIMEFRAME)
  #print(f"{result}")
  listcheck=[]
  user_id=2
  order_id=12
  exchange_id=1
  amount=100
  #full_logic(symbol,listcheck)
  listcheck=full_logic(symbol,listcheck,user_id, order_id, exchange_id, amount)
  #all_exited= 0
  #target_time= binance2.get_order( symbol, 17495)
  #target_time= binance2.create_slm_order("buy", symbol, 0.6,620)
  #target_time= binance2.modify_stop_loss( symbol, order_id=6851857, new_stop_price=700, price=680, quantity=0.1)
  #target_time,all_exited= binance2.modify_sl_order_for_sell(symbol, 17468, 590, 0.4,all_exited)
  #target_time=binance2.modify_mkt_order( 6853385, 0.5, symbol)
  print(listcheck)
  
  
  
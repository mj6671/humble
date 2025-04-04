import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import datetime
from dotenv import load_dotenv
import os
from models.model import fetch_pending_orders, fetch_user_api_keys, save_order_to_db, update_users_orders,log_trade,update_order_status
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

BUY_ONLY_CALL_OR_PUT=os.getenv("BUY_ONLY_CALL_OR_PUT")

Isl_percentage = os.getenv("Isl_percentage", "10%")  

length = os.getenv("length")
factor = os.getenv("factor")

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


#binance = ExchangeFactory.get_exchange_client("1", api_keys["api_key"], api_keys["secret_key"])
# Function to fetch OHLCV data
def fetch_ohlcv(symbol):
    bars = binance.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    #print(f"red {df}")  # Proper indentation
    
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
    ticker = binance.fetch_ticker(symbol)  
    last_price = ticker['last']
    if last_price is None:
        raise ValueError(f"Error: Could not fetch last price for {symbol}")
    return last_price  # Ensure return is properly aligned


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
	
def calculate_last_bearish_candle(df):
        df = df.iloc[::-1]
    
        # find the first row with 'open' value greater than 'close'
        for row in df.itertuples():
            if row.open > row.close:
                result = row.low
                return result	
				
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
            if individual_values[f'supertrend_{i}'] == 'yes':
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


def full_logic(ce_tradingsymbol,listcheck):
            global opt_tf
            
            opt_tf= TIMEFRAME
           
				
            selected_values=(candle_condition, previous_candle,close_condition,
                    condition,call_candle_condition,
                    call_previous_candle,call_close_condition,
                    call_condition,put_candle_condition,
                    put_previous_candle,put_close_condition,
                    put_condition,ce_tradingsymbol)##last strike price from premium pending
            
            call_trade,put_trade,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df = create_trade_conditions(*selected_values,opt_tf,listcheck)
            print('new',call_trade,put_trade,supertrend_value)
            
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
            order = binance.create_market_buy_order(symbol, amount)
            price = order["fills"][0]["price"] if order.get("fills") else order["price"]
            exchange_order_id= order["id"] 
            log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, "Simple Buy", price, amount, "executed")
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
            order = binance.create_market_sell_order(symbol, amount)
            price = order["fills"][0]["price"] if order.get("fills") else order["price"]
            exchange_order_id = order["id"]
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
	
  #df= fetch_ohlcv("BTCUSDT")
  #get_isl(df,2015)
  #get_last_bearish_candle(df,2015.50)
  #RSI,prev_rsi,rsi_df = get_rsi_df(df,RSI_LENGTH)
  symbol="ETHUSDT"
  #CC1,CC2,CO2,CH2,CL2,CRSI,cdf,prev_crsi,callp1_time,ce_rsi_df = get_data(symbol, TIMEFRAME, RSI_LENGTH)
 
  #dd=calculate_supertrend(cdf, int(length), int(factor))
 
  #latest_trend = dd["trend"].iloc[-1]
  #result = data_analysis(symbol,TIMEFRAME)
  #print(f"{result}")
  listcheck=[]
  full_logic(symbol,listcheck)
  
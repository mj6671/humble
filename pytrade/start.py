import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import os
import threading
import datetime

import time
import math
import sys
#import orderplacement

# Load environment variables from .env file
load_dotenv()

pd.options.mode.chained_assignment = None

# Fetch API credentials from environment variables
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
supertrend_length = os.getenv('SUPRETREND_LENGTH')
supretrend_factor = os.getenv('SUPRETREND_FACTOR')
ISL_PERCENTAGE=os.getenv('ISL_PERCENTAGE')
UPPER_LIMIT=os.getenv('UPPER_LIMIT')
LOWER_LIMIT=os.getenv('LOWER_LIMIT')
TSL1=os.getenv('TSL1')
TSL2=os.getenv('TSL2')
INDEX_CHART=os.getenv('INDEX_CHART')
CHART_REFERENCE=os.getenv('CHART_REFERENCE')
THIRD_CHART_REFERENCE=os.getenv('THIRD_CHART_REFERENCE')
BUY_ONLY_CALL_OR_PUT=os.getenv('BUY_ONLY_CALL_OR_PUT')
SUPERTREND = os.getenv('SUPERTREND')
call_time_frame=os.getenv('call_time_frame')
bnf_time_frame=os.getenv('bnf_time_frame')
opt_time_frame=os.getenv('opt_time_frame')
premium_limit=os.getenv('premium_limit')
ORDER_TYPE=os.getenv('ORDER_TYPE')

if not api_key or not api_secret:
    raise ValueError("API key and secret must be set in the .env file.")

# Set up API connection
client = Client(api_key, api_secret)

# Function to fetch historic data
def get_historic_data(token, timeframe):
    # Use Binance API to fetch historic data
    klines = client.get_klines(symbol=token, interval=timeframe)
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    return df

# Function to fetch current price
def get_ltp(token):
    # Use Binance API to fetch current price
    ticker = client.get_ticker(symbol=token)
    return float(ticker['lastPrice'])

# Function to fetch current price and high price
def get_ltp_and_current_opt_high(token, opt_tf):
    # Use Binance API to fetch current price and high price
    ticker = client.get_ticker(symbol=token)
    return float(ticker['lastPrice']), float(ticker['highPrice'])



def calculate_last_bearish_candle_with_rsi(df,lower_limit_value):
            df = df.iloc[::-1]
            for row in df.itertuples():
                if (row.open > row.close) and (row.rsi <lower_limit_value):
                    result = row.low
                    return result
def calculate_last_bearish_candle(df):
        df = df.iloc[::-1]
    
        # find the first row with 'open' value greater than 'close'
        for row in df.itertuples():
            if row.open > row.close:
                result = row.low
                return result
                                
    

# Function to create Heikin Ashi candles
def heikin_ashi_df(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (ha_df['open'].astype(float) + ha_df['high'].astype(float) + ha_df['low'].astype(float) + ha_df['close'].astype(float)) / 4
    
    # Initialize the first HA_Open
    ha_df['HA_Open'] = (ha_df['open'].astype(float) + ha_df['close'].astype(float)) / 2
    
    # Calculate HA_High and HA_Low
    ha_df['HA_High'] = ha_df[['open', 'close', 'high']].astype(float).max(axis=1)
    ha_df['HA_Low'] = ha_df[['open', 'close', 'low']].astype(float).min(axis=1)

    # Adjust subsequent HA_Open values
    for i in range(1, len(ha_df)):
        ha_df.at[i, 'HA_Open'] = (ha_df.at[i-1, 'HA_Open'] + ha_df.at[i-1, 'HA_Close']) / 2

    return ha_df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]

# Function to calculate and log Heikin-Ashi data
def heikin_ashi(rel_df):
    df = heikin_ashi_df(rel_df)
    last_2_rows = df.tail(2)
    
    C1 = last_2_rows.iloc[0]['HA_Close']
    C2 = last_2_rows.iloc[1]['HA_Close']
    O2 = last_2_rows.iloc[1]['HA_Open']
    H2 = last_2_rows.iloc[1]['HA_High']
    L2 = last_2_rows.iloc[1]['HA_Low']
    P1_time = rel_df.tail(2).iloc[1]['date']
    
    last_row = last_2_rows.iloc[1][['HA_Open', 'HA_Close']]
    RO = round(last_row.mean(), 2)
    
    return C1, C2, O2, H2, L2, RO, df, P1_time

def calculate_last_rsi(df, period=14):
    delta = df['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up = up.ewm(com=period-1, adjust=False).mean()
    roll_down = down.ewm(com=period-1, adjust=False).mean().abs()

    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))

    return RSI.iloc[-1]

def get_data(name, timeframe, rsi_length):
    new_data_df = get_historic_data(name, timeframe)
    
    # Calculate Heikin-Ashi values
    C1, C2, O2, H2, L2, RO, hdf, p1_time = heikin_ashi(new_data_df)
    
    # Calculate RSI values
    RSI,prev_rsi,rsi_df = calculate_last_rsi(new_data_df, rsi_length)
    
    return C1,C2,O2,H2,L2,RSI,hdf,prev_rsi,p1_time,rsi_df

def last_2_candles_low_sl(sl_df):
       
        sl_df= sl_df.tail(2)
        first_row_low = sl_df.iloc[0]['low']
        second_row_low = sl_df.iloc[1]['low']
    
        print("Low value of the first row:", first_row_low)
        print("Low value of the second row:", second_row_low)
      
        lowest_low = sl_df['low'].min()
        print(f"Last 2 candles' lowest low: {lowest_low}, Time: {datetime.datetime.now()}")
        
    
        return lowest_low
def last_bearish_candle_low(name, timeframe,):
        df= get_historic_data(name, timeframe)
        C1,C2,O2,H2,L2,RO,hdf,p1_time = heikin_ashi(df)
          
        df = hdf.iloc[::-1]
        for row in df.itertuples():
            if row.open > row.close:
                result = row.low
                return result
        
def get_close_and_rsi(name, timeframe,rsi_length):
        new_data_df= get_historic_data(name, timeframe)
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df)
        RSI, previous_rsi,rsi_df= calculate_last_rsi(new_data_df,rsi_length)
        print('get close and rsi:',datetime.datetime.now() )
             
        return C2,O2,RSI,previous_rsi,L2,p1_time

def bullish_condition(bull_condt, i, order_executed_price, trade, local_quantity, all_exited, sl_order_id, isl_value, name, timeframe, rsi_length):
    # Fetch historical data and apply Heikin Ashi transformation
    new_data_df = get_historic_data(name, timeframe)
    df = heikin_ashi_df(new_data_df)
    
    # Consider only the last three rows (candles)
    bullish_rows = df.tail(3).reset_index(drop=True)
    print(f"Bullish condition for candle {bull_condt}")
    print(bullish_rows)
    
    # Check if all three rows are bullish (close > open)
    are_bullish = all(bullish_rows['HA_Close'] > bullish_rows['HA_Open'])
    print(f"Three continuous bullish candles satisfied: {are_bullish}")
    
    if are_bullish:
        print("Three previous candles are continuously bullish")
        
        # Log the open and close values of the last three candles
        for idx, row in bullish_rows.iterrows():
            print(f"Candle {3-idx}: Open={row['HA_Open']}, Close={row['HA_Close']}")
        
        # Use hardcoded values for testing instead of pulling from Excel or a file
        hardcoded_value_p3 = bullish_rows.iloc[0]['HA_Close']  # Replace with desired hardcoded value if needed
        hardcoded_value_p2 = bullish_rows.iloc[1]['HA_Close']  # Replace with desired hardcoded value if needed
        hardcoded_value_p1 = bullish_rows.iloc[2]['HA_Close']  # Replace with desired hardcoded value if needed
        
        # Print selected values for each row
        print(f"Selected value for p3 candle: {hardcoded_value_p3}")
        print(f"Selected value for p2 candle: {hardcoded_value_p2}")
        print(f"Selected value for p1 candle: {hardcoded_value_p1}")

        # Check if any of the selected values are less than the order executed price
        if hardcoded_value_p3 < order_executed_price or hardcoded_value_p2 < order_executed_price or hardcoded_value_p1 < order_executed_price:
            print("Chosen value is lesser than order executed price. Not executing bullish condition logic.")
            return sl_order_id, isl_value, all_exited, bull_condt
        else:
            # Determine the lowest low among the three candles
            lowest_low = bullish_rows['HA_Low'].min()
            print(f"The lowest value in the 'low' column is: {lowest_low}")
            
            sl_price = round(float(lowest_low), 1)
            price = round(float(lowest_low - 0.1), 1)
            print(f"Bullish condition SL price is: {sl_price}")
            
            if bull_condt and isl_value < lowest_low:
                bull_condt = False
                # Modify stop-loss order
                
                sl_order_id = orderplacement.modify_sl_order(sl_order_id, sl_price, price, local_quantity)
                print("Shifting ISL/TSL value.")
                isl_value = lowest_low
            
            return sl_order_id, isl_value, all_exited, bull_condt
    else:
        print("Three previous candles aren't continuously bullish.")
        # Log the open and close values for the last three candles
        for idx, row in bullish_rows.iterrows():
            print(f"Candle {3-idx}: Open={row['HA_Open']}, Close={row['HA_Close']}")
        
        return sl_order_id, isl_value, all_exited, bull_condt

def time_calc(i,opt_tf,entry):
        
        # now = datetime.datetime.now()
        now = datetime.datetime.now().replace(microsecond=0)
    
        # seconds_since_start_of_minute = now.second
        if not entry:
            start_time = datetime.datetime(now.year, now.month, now.day, 9, 15, 4)
        else:
            start_time = datetime.datetime(now.year, now.month, now.day, 9, 15, 0)


        # intervals = {
        #     'minute': [start_time + datetime.timedelta(minutes=i) for i in range(0, 375)],
        #     '3minute': [start_time + datetime.timedelta(minutes=i*3) for i in range(0, 125)],
        #     '5minute': [start_time + datetime.timedelta(minutes=i*5) for i in range(0, 75)],
        #     '10minute': [start_time + datetime.timedelta(minutes=i*10) for i in range(0, 38)],
        #     '30minute': [start_time + datetime.timedelta(minutes=i*30) for i in range(0, 13)],
        #     '15minute': [start_time + datetime.timedelta(minutes=i*15) for i in range(0, 25)],
        #     '60minute': [start_time + datetime.timedelta(hours=i) for i in range(1, 7)]
        # }
        intervals = {
            'minute': [start_time + datetime.timedelta(minutes=i) for i in range(0, 375)],
            '2minute': [start_time + datetime.timedelta(minutes=i*2) for i in range(0, 188)],
            '3minute': [start_time + datetime.timedelta(minutes=i*3) for i in range(0, 125)],
            '4minute': [start_time + datetime.timedelta(minutes=i*4) for i in range(0, 94)],
            '5minute': [start_time + datetime.timedelta(minutes=i*5) for i in range(0, 75)],
            '10minute': [start_time + datetime.timedelta(minutes=i*10) for i in range(0, 38)],
            '30minute': [start_time + datetime.timedelta(minutes=i*30) for i in range(0, 13)],
            '15minute': [start_time + datetime.timedelta(minutes=i*15) for i in range(0, 25)],
            '60minute': [start_time + datetime.timedelta(hours=i) for i in range(1, 7)],
            '2hour': [start_time + datetime.timedelta(hours=i*2) for i in range(0, 12)],
            '3hour': [start_time + datetime.timedelta(hours=i*3) for i in range(0, 8)]
        }
    
        if opt_tf != '60minute' and datetime.datetime(now.year, now.month, now.day, 9, 15, 1) <= now <= datetime.datetime(now.year, now.month, now.day, 9, 15, 3):
            return intervals[opt_tf][1]
        
        target_times = intervals[opt_tf]
        for t in target_times:
            if now < t:
                return t
    
        return None 

def checking_complete_status(opt_low,isl_value,bearish_sl,all_exited,local_quantity,qty,status,sl_order_id,name):
    
        if (status =='COMPLETE' and local_quantity==0):    
            all_exited= True
            
            local_quantity=0
            return all_exited,sl_order_id,local_quantity,status,isl_value
        
        elif (status=='COMPLETE' and local_quantity!=0):
            if bearish_sl:
                # sl=last_bearish_candle_low(name, opt_tf, 4, 'NFO',logFile)
                print("last bearish candle sl is:",opt_low)
                
                qty=local_quantity
                sl_price= round(float(opt_low),1)
                price= round(float(opt_low-0.1),1)
                isl_value= opt_low
                ltp= get_ltp(name)
    
                if sl_price>ltp:
                    print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                                      
                    order_type= 'market'
                
                    limit_value=0                                 
                    oid=orderplacement.order_placement(name,'SELL',qty,order_type,limit_value)
                    status = orderplacement.get_open_order_details(oid)
                
                    if status=="COMPLETE":
                        all_exited=True
                        
                        return all_exited,sl_order_id,local_quantity,status,isl_value
    
                else:
             
                    sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(all_exited,name,qty,price,sl_price)        
                    
                    print("sl order id", sl_order_id)            
                    print('bearish candle sl is', sl_price)
           
                    return all_exited,sl_order_id,local_quantity,status,isl_value
            
    
        else:
            return all_exited,sl_order_id,local_quantity,status,isl_value

def check_candle_condition(opt_low,trade,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name):
    
        if status=='COMPLETE':
            all_exited,sl_order_id,local_quantity,status,isl_value= checking_complete_status(opt_low,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,status,sl_order_id,name)        
            return all_exited,sl_order_id,local_quantity,isl_value
        
        elif status!='COMPLETE':
            if (status== 'REJECTED' or status=='CANCELLED'):
                print("condition met but order placement failed due to some reasons placing orders again")
                  
                for i in range(1,100):
                    oid= orderplacement.modify_mkt_order(sl_order_id,qty)
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                    all_exited,sl_order_id,local_quantity,status,isl_value= checking_complete_status(opt_low,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name)        
                    if status=='COMPLETE':
                        return all_exited,sl_order_id,local_quantity,isl_value
                    else:              
                        time.sleep(1)
    
                local_quantity= local_quantity+qty                
                return all_exited,sl_order_id,local_quantity,isl_value
                        
            elif status=='PENDING': 
                print("condition met but order placement pending due to some reasons checking again")
               
                for i in range(1,100):
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                    all_exited,sl_order_id,local_quantity,status,isl_value= checking_complete_status(opt_low,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name)        
                    if status=='COMPLETE':
                        return all_exited,sl_order_id,local_quantity,isl_value
                    else:              
                        time.sleep(1) 
    
                local_quantity= local_quantity+qty
                return all_exited,sl_order_id,local_quantity,isl_value
def qty_to_exit(name,logFile,i,local_quantity):
            print("current qty logic")
            cq= local_quantity
           
            if individual_values[f'index_{i}']=='BANKNIFTY':
                lot_size=individual_values[f'bnf_lot_size_{i}']
            elif individual_values[f'index_{i}']=='NIFTY':
                lot_size=individual_values[f'nifty_lot_size_{i}']
            elif individual_values[f'index_{i}']=='FINNIFTY':
                lot_size=individual_values[f'finnifty_lot_size_{i}']
            elif individual_values[f'index_{i}']=='SENSEX':
                lot_size=individual_values[f'sensex_lot_size_{i}']
            elif individual_values[f'index_{i}']=='BANKEX':
                lot_size=individual_values[f'bankex_lot_size_{i}']
            elif individual_values[f'index_{i}']=='MIDCPNIFTY':
                lot_size=individual_values[f'midcapnifty_lot_size_{i}']
           
            lots=cq/lot_size
    
            if lots%2 ==0:           
                quant_to_sell= int(cq/2)
                print('quant_to_sell', quant_to_sell)
                 
                local_quantity=local_quantity-quant_to_sell          
    
                return quant_to_sell,local_quantity
            else:
                lots_to_sell = math.ceil(lots / 2)
                quant_to_sell= int(lots_to_sell*lot_size)
                print('quant_to_sell', quant_to_sell)
                 
                local_quantity=local_quantity-quant_to_sell                    
    
                return quant_to_sell,local_quantity

def EMA(df, base, target, period, alpha=False):
        con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

        if (alpha == True):
            # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
            df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
        else:
            # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
            df[target] = con.ewm(span=period, adjust=False).mean()

        df[target].fillna(0, inplace=True)
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

def SuperTrend(df, period, multiplier, ohlc=['open', 'high', 'low', 'close']):
        print('supertrend_period & supertrend_multiplier:','period & multiplier:',period,multiplier)
        ATR(df, period, ohlc=ohlc)
        atr = 'ATR_' + str(period)
        st = 'ST' #+ str(period) + '_' + str(multiplier)
        stx = 'STX' #  + str(period) + '_' + str(multiplier)

        # Compute basic upper and lower bands
        df['basic_ub'] = ((df[ohlc[1]] + df[ohlc[2]]) / 2) + multiplier * df[atr]
        df['basic_lb'] = ((df[ohlc[1]] + df[ohlc[2]]) / 2) - multiplier * df[atr]

        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                            df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else \
            df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                            df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else \
            df['final_lb'].iat[i - 1]

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[
                i] <= df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] > \
                                        df['final_ub'].iat[i] else \
                    df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= \
                                            df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] < \
                                                df['final_lb'].iat[i] else 0.00

            # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down', 'up'), np.NaN)

        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)
        return df

def get_supertrend(df):
        # df=get_historic_data(k_ts,bnf_tf,delta,itype,logFile,5)
        # hdf=heikin_ashi_df(df,logFile)
        supertrend_period = supertrend_length
        supertrend_multiplier = supretrend_factor
        sdf = SuperTrend(df,int(supertrend_period),int(supertrend_multiplier))
        df= sdf.tail(1)
        supertrend_value= df['STX'].values[0]
        print('super trend is',supertrend_value)
        return supertrend_value


def data_analysis(i,index_future,pe_tradingsymbol,ce_tradingsymbol,bnf_tf,opt_tf):
                
        ce_ts=ce_tradingsymbol
        pe_ts=pe_tradingsymbol
        k_ts= index_future
        rsi_length= os.getenv('RSI_LENGTH')
        #print("rsi length is",rsi_length)  
        #print('symbols', ce_ts,pe_ts,k_ts)
      
        CC1,CC2,CO2,CH2,CL2,CRSI,cdf,prev_crsi,callp1_time,ce_rsi_df= get_data(ce_ts,opt_tf,4,'NFO',rsi_length)               
        PC1,PC2,PO2,PH2,PL2,PRSI,pdf,prev_prsi,putp1_time,pe_rsi_df= get_data(pe_ts,opt_tf,4,'NFO',rsi_length)
        C1,C2,O2,H2,L2,RSI,fdf,prev_rsi,candle_p1_time,fut_rsi_df= get_data(k_ts,bnf_tf,4,'NFO',rsi_length)
        supertrend_value=get_supertrend(fdf)
        call_supertrend_value=get_supertrend(cdf)
        put_supertrend_value=get_supertrend(pdf)

        print("Candle data Extration completion time",datetime.datetime.now())

        return C1,C2,O2,H2,L2,RSI,prev_rsi,candle_p1_time,CC1,CC2,CO2,CH2,CL2,CRSI,prev_crsi,callp1_time,PC1,PC2,PO2,PH2,PL2,PRSI,prev_prsi,putp1_time,cdf,pdf,supertrend_value,call_supertrend_value,put_supertrend_value,ce_rsi_df,pe_rsi_df,fut_rsi_df
       
def get_ltp(tradingsymbol):
        ltp=client.get_symbol_ticker(symbol=tradingsymbol)
        #print('ltp is:',ltp['price'])
        ltp= ltp['price']
        return ltp

def get_ltp_and_current_opt_open(tradingsymbol,opt_tf):
        ltp=get_ltp(tradingsymbol)
       # print('Inside get_ltp_and_current_opt_open: ltp is:',ltp,opt_tf)
       
    
        new_data_df= get_historic_data(tradingsymbol, opt_tf)
             
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df)
        #print('returning ltp for trade time',datetime.datetime.now())
             
        return ltp,RO  

def get_ltp_and_current_opt_high(tradingsymbol,opt_tf):
        ltp=get_ltp(tradingsymbol)
      #  print('Inside get_ltp_and_current_opt_high: ltp is:',ltp,opt_tf)
       
        new_data_df= get_historic_data(tradingsymbol, opt_tf)    
       
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df)        
       # print('returning ltp for trade time',datetime.datetime.now())
      
        
        return ltp,H2 

def target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,order_exec_price,all_exited,sl_order_id,local_quantity):
        TARGET_SPLIT=os.getenv('TARGET_SPLIT')
        FIRST_TARGET=os.getenv('FIRST_TARGET')
        FIRST_TARGET_TRAILING=os.getenv('FIRST_TARGET_TRAILING')
        SECOND_TARGET=os.getenv('SECOND_TARGET')
        SECOND_TARGET_TRAILING=os.getenv('SECOND_TARGET_TRAILING')
        TARGET=os.getenv('TARGET')
        
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
                    first_qty,local_quantity=qty_to_exit(name,i,local_quantity)
                    first_target_percent= FIRST_TARGET_TRAILING
                    print("first target percent is", first_target_percent)
                    total_point=target*first_target_percent
                    print("total_point", total_point)

                    first_target_sl= target-total_point
                    print("first target sl is",first_target_sl)
                    sl_price= round(float(first_target_sl),1)
                    isl_value=sl_price
                    price= round(float(first_target_sl-0.1),1) 
                    sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,first_qty,all_exited)                    
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
                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(sl_order_id)
                print("first targetr order status is not completed entering into while loo[p]")
                first_target_percent= FIRST_TARGET_TRAILING
                while status !="COMPLETE":
                    o_ltp=get_ltp(name)
                    if o_ltp > first_target_trailing_point:
                        
                        isl_value=isl_value+total_point
                        sl_price= round(float(isl_value),1)
                        price= round(float(isl_value-0.1),1) 
                        sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,first_qty,all_exited)
                        if all_exited:
                            second_target=True
                            all_exited=False
                            first_target_trailing=False
                            first_target=False
                            break

                        first_target_trailing_point=first_target_trailing_point+total_point
                    
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(sl_order_id)
                    
                print("first targetr order status is completed entering into second target")
                second_target=True
                all_exited=False
                first_target_trailing=False
                first_target=False
                second_sl=True


            if second_target:
                if second_sl:
                    carry= individual_values[f'product_type_{i}']
                    if carry=='normal':
                        product_type=kite.PRODUCT_NRML
                    elif carry=='intraday':
                        product_type=kite.PRODUCT_MIS                    
                    sl_price= round(float(order_exec_price),1)
                    price= round(float(order_exec_price-0.1),1)
                    sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(carry,all_exited,name,local_quantity,kite.ORDER_TYPE_SL,product_type,price,kite.TRANSACTION_TYPE_SELL,kite.EXCHANGE_NFO,kite.VARIETY_REGULAR,sl_price,kite.VALIDITY_DAY,logFile,kite,Order_Identifier)        
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
                    sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)                    
                    second_target_trailing =True
                    second_target_trailing_point= target + (target*SECOND_TARGET_TRAILING)
                    print("second target trailing point is",second_target_trailing_point)
                    
                    print("sl has been modified for half quantity")
                    
                                       
                    second_target=False
            
                    if all_exited:
                        second_target_trailing=False
                        all_exited=True
                        
            
            if second_target_trailing:
                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(sl_order_id)
                print("second targetr order status is not completed entering into while loo[p]")
                second_target_percent= SECOND_TARGET_TRAILING
                while status !="COMPLETE":
                    o_ltp=get_ltp(name)
                    if o_ltp > second_target_trailing_point:
                        
                        isl_value=isl_value+total_point
                        sl_price= round(float(isl_value),1)
                        price= round(float(isl_value-0.1),1) 
                        sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                        if all_exited:
                            
                            all_exited=True
                            
                            break

                        second_target_trailing_point=second_target_trailing_point+total_point
                    
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',sl_order_id)
                    
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
                oid= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
                time.sleep(5)
                #status= orderplacement.get_orderstatus(oid,logFile)
                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_xec_price,exchange_timestamp1 = orderplacement.get_open_order_details(oid)
            
                if status=='COMPLETE':
                    all_exited=True
                    
                    local_quantity=0

                else:
                    print("status is not complete",status)
            
            return all_exited,local_quantity,first_target,second_target,sl_order_id
            
        return all_exited,local_quantity,first_target,second_target,sl_order_id
     
def single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl):
        print("entered into single candle condition checking")
        print("tsl movement is", tslmovement)
            
        no_of_times= os.getenv('SINGLE_CANDLE_CONDITION')  ##get the values from SINGLE_CANDLE_CONDITION
        o_ltp,current_candle_open=get_ltp_and_current_opt_open(name,opt_tf)
        single_candle_target= (no_of_times*current_candle_open) + current_candle_open
        print('no of times:',no_of_times, 'o_ltp;',o_ltp, 'single candle target', single_candle_target, 'current candle open',current_candle_open)
        
        
        ###target checking###
        if o_ltp> single_candle_target:
            print("single candle target achieved")
          
            selected_percent= os.getenv('TSL_OF_SCC')  ##get the values from TSL_OF_SCC
            single_percent_sl= selected_percent * current_candle_open
            single_candle_sl= single_candle_target-single_percent_sl
            print("single candle sl",single_candle_sl)
            qty=local_quantity
            if single_candle_sl!=ssl:
                ssl=single_candle_sl
                if single_candle_sl > isl_value:            
                    sl_order_id,all_exited= orderplacement.def_modify_sl_order_for_sell(sl_order_id,single_candle_sl,qty,all_exited)
                    isl_value=single_candle_sl            
                
            if not tslmovement:
                print("if not tslmovement")
               
                trailing_sl_movement_percent= os.getenv('after_scc_x_pct_price_move')   ##get the values from excel
                trailing_sl_movement= (trailing_sl_movement_percent * current_candle_open)
                trailing_sl_target= single_candle_target+ trailing_sl_movement
                tsl_percent=  os.getenv('after_scc_y_pct_trailing_move')   ##get the values from excel
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
                    sl_order_id,all_exited= orderplacement.def_modify_sl_order_for_sell(sl_order_id,tsl,qty,all_exited) 
                    isl_value=tsl
                    
                    
                else:
                    print(f'single candle sl shifting already happened and {tsl} value is much closer to ltp so tsl not happened')     
                                     
         
                trailing_sl_target= trailing_sl_target+trailing_sl_movement
        
        print(" trailing sl revised target",trailing_sl_target)
            
        return sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value

def logic_checking(opt_low,trade,isl_value,logFile,i,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited):
        print("entered into logic_checking")
        # qty,local_quantity = qty_to_exit(name,logFile,i,local_quantity)    
        # print(' exiting 50 percent at market:', qty,'Available quantity is',local_quantity)
        # if individual_values[f'index_{i}']=='BANKNIFTY':
        #     lot_size=individual_values[f'bnf_lot_size_{i}']
        # if individual_values[f'index_{i}']=='NIFTY':
        #     lot_size=individual_values[f'nifty_lot_size_{i}']
        # if individual_values[f'index_{i}']=='FINNIFTY':
        #     lot_size=individual_values[f'finnifty_lot_size_{i}']
        TSL1=os.getenv('TSL1')
        TSL2=os.getenv('TSL2')
       
        if (TSL1== 'Exit half and remaining at bearish candle low' 
            or TSL2== 'Exit half and remaining at bearish candle low' 
            or TSL1== 'Exit at market price and half trailing with bearish candle low sl'
            or TSL2== 'Exit at market price and half trailing with bearish candle low sl'
            or TSL1== 'Full trailing and half trailing with bearish candle low sl'
            or TSL2== 'Full trailing and half trailing with bearish candle low sl'):
            bearish_sl=True
            print("entered into Exit half and remaining at bearish candle low")
            
            # oid= orderplacement.modify_mkt_order(sl_order_id,qty)
            # order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
            # all_exited,sl_order_id,local_quantity,isl_value= check_candle_condition(opt_low,trade,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name,opt_tf,logFile)
            sl_price= round(float(opt_low),1)
            price= round(float(opt_low-0.1),1)
            sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
            

            isl_value=opt_low
            
            return all_exited,sl_order_id,local_quantity,isl_value

def checking_condition(trade,ssl,last_candle_close,order_exec_price,i,name,opt_tf,logFile,sl_order_id,all_exited,local_quantity,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl):
        print("entered into exit at market checking conditionm")
       
    
        if last_candle_close < order_exec_price: ##exit all quantity logic
            print("p1_candle_low < order_exec_price ",'last low is',last_candle_close,'order_executed price is',order_exec_price)
            print(" condition satisfied closing the trade")
          
    
            exit_all= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
            #status=orderplacement.get_orderstatus(exit_all,logFile)
            order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',exit_all)
            if status=='COMPLETE':
                all_exited= True
                print("exited all")
                
                
    
                local_quantity=0
                return all_exited,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,local_quantity,ssl
                                           
                
    
        if True and not all_exited:
            sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
            o_ltp=get_ltp(name,logFile)
            ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)

            all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)
        
        
        return all_exited,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,local_quantity,ssl
    
def get_qty(symbol):
    """
  quantity (float): Desired quantity of the asset to be bought or sold.

    Returns:
        float: The adjusted quantity to be used for the order, complying with Binance filters.
    """
    CBT=os.getenv('CBT')
    CAPITAL_TO_DEPLOY= float(os.getenv('CAPITAL_TO_DEPLOY', 0))
    QBT=os.getenv('QBT')
    QUANTITY=float(os.getenv('QUANTITY', 0))
    
     # Determine amount or desired quantity
    amount = CAPITAL_TO_DEPLOY if CBT == 'yes' else None
    desired_quantity = QUANTITY if QBT == 'yes' else None
        
     # Fetch the latest price for the symbol
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
       
    except BinanceAPIException as e:
        raise ValueError(f"Error fetching price for {symbol}: {e}")

    # Fetch symbol information to get the filters
    try:
        symbol_info = client.get_symbol_info(symbol)
        filters = {f['filterType']: f for f in symbol_info['filters']}
    except BinanceAPIException as e:
        raise ValueError(f"Error fetching symbol info for {symbol}: {e}")

    # Extract filter values
    min_qty = float(filters['LOT_SIZE']['minQty'])
    max_qty = float(filters['LOT_SIZE']['maxQty'])
    step_size = float(filters['LOT_SIZE']['stepSize'])

    # Calculate quantity based on desired_quantity or amount
    if desired_quantity is not None:
        quantity = desired_quantity
    elif amount is not None:
        quantity = amount / price
    else:
        raise ValueError("Either amount or desired_quantity must be provided.")

    # Adjust quantity to comply with the step size
    quantity = quantity - (quantity % step_size)

    # Ensure the quantity is within Binance limits
    if quantity < min_qty:
        raise ValueError(f"Quantity {quantity} is below the minimum allowed quantity of {min_qty} for {symbol}.")
    elif quantity > max_qty:
        raise ValueError(f"Quantity {quantity} exceeds the maximum allowed quantity of {max_qty} for {symbol}.")

    return quantity

def get_isl(df,order_exec_price):
        sl=last_2_candles_low_sl(df) 

       
        print('last two candles low:',sl)
        print('order exec price:', order_exec_price)
        if sl< (order_exec_price - (ISL_PERCENTAGE*order_exec_price)):
            sl= order_exec_price - (ISL_PERCENTAGE *order_exec_price)
        
        return sl

def get_last_bearish_candle(df,order_exec_price):
        sl= calculate_last_bearish_candle(df)
        print('last bearish candle low is',sl)
      

        if sl< (order_exec_price - (ISL_PERCENTAGE *order_exec_price)):
            sl= order_exec_price - (ISL_PERCENTAGE *order_exec_price) 
        return sl

def get_last_bearish_candle_with_rsi(df,order_exec_price,lower_limit_value):
        sl= calculate_last_bearish_candle_with_rsi(df,lower_limit_value)
        print('last bearish candle low is',sl)


        if sl< (order_exec_price - (ISL_PERCENTAGE *order_exec_price)):
            sl= order_exec_price - (ISL_PERCENTAGE *order_exec_price) 
        return sl
def full_trailing_checking(trade,opt_low,isl_value,sl_order_id,all_exited,local_quantity):
        if opt_low > isl_value:
            sl_price= round(float(opt_low),1)
            price= round(float(opt_low-0.1),1) 
            sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
            
            

            isl_value=opt_low
            print("Shifting isl/tsl value")
            print("New tsl is",isl_value)
         
        return sl_order_id,isl_value,all_exited

def check_tsl(prev_rsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity):
        
      
        
        if (((call_trade_taken and prev_rsi>rsi and (orsi < UPPER_LIMIT 
                )) or (put_trade_taken and rsi>prev_rsi
                and (orsi < UPPER_LIMIT)))
                and (not entered_condition) 
                and ts1_condition_satisfied 
                and (TSL1== 'Exit at market price and half trailing with bearish candle low sl' or TSL1== 'Exit half and remaining at bearish candle low')) :
        
        
                                           
            print(f"given {TSL1} condition satisfied for the candle time of {last_candle_time}")
          
            entered_condition = True
            all_exited, sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  

        
        elif ( ((call_trade_taken and prev_rsi>rsi
                and  orsi< LOWER_LIMIT)
                or (put_trade_taken and rsi>prev_rsi
                and  orsi< LOWER_LIMIT))
                and (not entered_condition)  and (TSL2== 'Exit at market price and half trailing with bearish candle low sl' or TSL2== 'Exit half and remaining at bearish candle low')):
        # if True:  
                print(f"given {TSL2} condition satisfied for the candle time of {last_candle_time}")                                                        
                
                
                entered_condition = True  
                all_exited,sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  

        else:
            print("TSL condition not satisfies so not taking any trades")
          

        return all_exited,sl_order_id,local_quantity,entered_condition,isl_value


def check_tsl_with_extra_condition(prev_rsi,trade,rsi,last_candle_time,opt_low,isl_value,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity):
        
        if (((call_trade_taken and prev_rsi>rsi and orsi < UPPER_LIMIT) 
                or (put_trade_taken and rsi>prev_rsi and orsi < UPPER_LIMIT)) 
                and not entered_condition
                and ts1_condition_satisfied 
                and TSL1== 'Full trailing and half trailing with bearish candle low sl') :
                                           
            print(f"given {TSL1} condition satisfied for the candle time of {last_candle_time}")                                                        
          
            
            entered_condition = True
            all_exited, sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  
        
        elif (((call_trade_taken and prev_rsi>rsi  and  orsi< LOWER_LIMIT)
                or (put_trade_taken and rsi>prev_rsi  and  orsi< LOWER_LIMIT))
                and not entered_condition  
                and ts2_condition_satisfied 
                and TSL2== 'Full trailing and half trailing with bearish candle low sl' ):
        # if True:  
                print(f"given {TSL2} condition satisfied for the candle time of {last_candle_time}")                                                        
               
                
                entered_condition = True  
                all_exited,sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  

        
        elif (opt_low > isl_value and not entered_condition):
                sl_price= round(float(opt_low),1)
                price= round(float(opt_low-0.1),1) 
                sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                

                isl_value=opt_low
                print("isl value is",isl_value)
              

        return all_exited, sl_order_id,local_quantity,entered_condition,isl_value


 ################ ENTRY CONDITION LOGIC FUNCTION   ###############

def create_trade_conditions(candle_condition, previous_candle, close_condition, condition, 
                        call_candle_condition, call_previous_candle, call_close_condition, call_condition,
                        put_candle_condition, put_previous_candle, 
                        put_close_condition, put_condition,index_future,pe_tradingsymbol,ce_tradingsymbol, i, bnf_tf,opt_tf,listcheck):
            
            
            print("entered create trade")
          
            while True:
                now= datetime.datetime.now()
                print('time is', now)                
                if 4 < now.second < 50:
                    break
                time.sleep(1)
            C1,C2,O2,H2,L2,RSI,prev_rsi,trade_taken_time,CC1,CC2,CO2,CH2,CL2,CRSI,prev_crsi,callp1_time,PC1,PC2,PO2,PH2,PL2,PRSI,prev_prsi,putp1_time,cdf,pdf,supertrend_value,call_supertrend_value,put_supertrend_value,ce_rsi_df,pe_rsi_df,fut_rsi_df= data_analysis(i,index_future,pe_tradingsymbol,ce_tradingsymbol, bnf_tf,opt_tf)
            #C1,C2,O2,H2,L2,RSI,prev_rsi,CC1,CC2,CO2,CH2,CL2,CRSI,prev_crsi,PC1,PC2,PO2,PH2,PL2,PRSI,prev_prsi,cdf,pdf,trade_taken_time
    
            UL=UPPER_LIMIT
            LL=LOWER_LIMIT
            #new_LL= individual_values[f'put_ll_{i}']
           # new_UL= individual_values[f'put_ul_{i}']

            new_LL= LOWER_LIMIT
            new_UL= UPPER_LIMIT
            print(new_LL,new_UL)
    
            print('Returning all the values time:',datetime.datetime.now())
            
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
            
            
         #   print("Third chart reference is",THIRD_CHART_REFERENCE)
          #  print("option chart or future chart is",INDEX_CHART)


            if INDEX_CHART== 'future_chart':
              #  print("Selected Future Chart")
                if THIRD_CHART_REFERENCE== 'yes':


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
                    print(conditions[0],conditions[1])

                    # Reconstruct the put condition string with the updated first two conditions
                    put_condition_str = f"({') and ('.join(conditions)})"
                    
                    

                    print("call condition string which includes third chart as well", call_condition_str)
                    print("put condition string which includes third chart as well", put_condition_str)

                elif THIRD_CHART_REFERENCE== 'no':

                    call_condition_str = f"({candle_conditions[candle_condition]} and {combined_condition}) and " \
                                f"({call_candle_conditions[call_candle_condition]} and {call_combined_condition})"
                    
                    put_condition_str = f"({candle_conditions[candle_condition]} and {combined_condition}) and " \
                                f"({put_candle_conditions[put_candle_condition]} and {put_combined_condition})"
                    
                                
                    put_condition_str = put_condition_str.replace('>', 'temp').replace('<', '>').replace('temp', '<')
                    put_condition_str = put_condition_str.replace('UL', 'temp').replace('LL', 'UL').replace('temp', 'LL')
                    conditions = put_condition_str.split(') and (')
                    print("conditions are",conditions)

                    # Replace the UL and LL in the first two conditions
                    conditions[0] = conditions[0].replace('LL', 'new_LL').replace('UL', 'new_UL')
                    print(conditions[0])
                    
                    # Reconstruct the put condition string with the updated first two conditions
                    put_condition_str = f"({') and ('.join(conditions)})"

                    print("call condition string which doesnt include third chart", call_condition_str)
                    print("put condition string which doesnt include third chart", put_condition_str)

            elif INDEX_CHART== 'option_chart':
                print("Selected Option Chart")


                if THIRD_CHART_REFERENCE== 'yes':


                    call_condition_str = f"({call_candle_conditions[call_candle_condition]} and {call_combined_condition}) and " \
                                f"({put_candle_conditions[put_candle_condition]} and {put_combined_condition})"
                                
            
            
                    put_condition_str = call_condition_str.replace('>', 'temp').replace('<', '>').replace('temp', '<')
                    put_condition_str = put_condition_str.replace('UL', 'temp').replace('LL', 'UL').replace('temp', 'LL')
                    conditions = put_condition_str.split(') and (')
                    print(conditions)

                    # Replace the UL and LL in the first two conditions
                    conditions[0] = conditions[0].replace('LL', 'new_LL').replace('UL', 'new_UL')
                    conditions[1] = conditions[1].replace('LL', 'new_LL').replace('UL', 'new_UL')
                    print(conditions[0],conditions[1])

                    # Reconstruct the put condition string with the updated first two conditions
                    put_condition_str = f"({') and ('.join(conditions)})"
                    
                    

                    print("call condition string which includes third chart as well", call_condition_str)
                    print("put condition string which includes third chart as well", put_condition_str)

                elif THIRD_CHART_REFERENCE== 'no':

                    call_condition_str = f"({call_candle_conditions[call_candle_condition]} and {call_combined_condition})"
                                
                    
                    put_condition_str = f"({put_candle_conditions[put_candle_condition]} and {put_combined_condition})"
                                
                    
                                
                    put_condition_str = put_condition_str.replace('>', 'temp').replace('<', '>').replace('temp', '<')
                    put_condition_str = put_condition_str.replace('UL', 'temp').replace('LL', 'UL').replace('temp', 'LL')
                    conditions = put_condition_str.split(') and (')
                    print("conditions are",conditions)

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
            print('\nCondition checking!:',print_str,'UL-->:',UPPER_LIMIT,'LL-->:',LOWER_LIMIT)
            print('CALL entry Conditions:',(call_condition_str),eval(call_condition_str))
            print('PUT entry Conditions:',(put_condition_str),eval(put_condition_str))
            
                    
            #exit()
            #print('Time after eval function',datetime.datetime.now() )
        

            if INDEX_CHART== 'future_chart':
    
                if BUY_ONLY_CALL_OR_PUT=="both":     
                    #time.sleep(60)
                    if SUPERTREND=='yes':
                        # previous_put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.
                        previous_put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.
                        previous_call_condition_met = eval(call_condition_str)
                        # previous_call_condition_met = eval(call_condition_str)
                        if previous_call_condition_met and supertrend_value=='up':
                            print("call condition met and super trend value is up so taking call trade")
                            call_condition_met=True
                            put_condition_met=None
                        elif previous_put_condition_met and supertrend_value=='down':
                            print("put condition met and super trend value is down so taking put trade")
                            put_condition_met=True
                            call_condition_met=None
                        else:
                            if previous_call_condition_met or previous_put_condition_met:
                                print("call or put conditions met but super trend conditions havent been met ")
                                put_condition_met=None
                                call_condition_met=None
                            else:
                                print("call and put condition was not satisfied in the first place")
                                put_condition_met=None
                                call_condition_met=None

                    else:
                        print("supertrend is not considered")
                        put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.
                        call_condition_met = eval(call_condition_str)
                        # print('call df', cdf)
                    # print('put df', pdf)
                    # exit() ## hardcoding needs to correct.
                    
                    print(f" call condition {call_condition_met} and put condition {put_condition_met}")
                    if call_condition_met:
                        print_str = "CALL CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                        print_str += f"Current Time: {datetime.datetime.now()}  \n"
                        print('CALL entry Conditions:',(call_condition_str),eval(call_condition_str))
                        print("Call conditions met and starting to take trade")
                  
                        signal=True
                        if signal:
                            t1= cdf['date'].iloc[-1]
                            t2= cdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                print('t1 and t2 not in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                             

                                listcheck.clear()
                                listcheck.append(t1)
                                listcheck.append(t2)
                                
                                call_condition_met=True
                                put_condition_met=None
                            else:
                                print("trade had benn already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                                call_condition_met=None
                                put_condition_met=None

                              


                        

                    elif put_condition_met:
                        print_str = "PUT CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                        print_str += f"Current Time: {datetime.datetime.now()}  \n"                    
                        print('PUT entry Conditions:',(put_condition_str),eval(put_condition_str))
                        print("Put conditions met and starting to take trade")
                                        
                        signal=True
                        if signal:
                            t1= pdf['date'].iloc[-1]
                            t2= pdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                listcheck.clear()
                                listcheck.append(t1)
                                listcheck.append(t2)
                                
                                call_condition_met=None
                                put_condition_met=True
                                print("Since there is no existing call trades, going to take put trade")
                            
                            else:
                                call_condition_met=None
                                put_condition_met=None 
                                print("trade had benn already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                            
                  
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df
                    
            
                elif BUY_ONLY_CALL_OR_PUT=="call":
                    if SUPERTREND=='yes':
                        
                        previous_call_condition_met = eval(call_condition_str)
                        if previous_call_condition_met and supertrend_value=='up':
                            print("call condition met and super trend value is up so taking call trade")
                            call_condition_met=True
                            put_condition_met=None
                        
                        else:
                            if previous_call_condition_met:
                                print("call condition met but super trend value is not up so not  taking call trade")

                                put_condition_met=None
                                call_condition_met=None
                            else:
                                print("call condition was not satisfied in the first place")
                                put_condition_met=None
                                call_condition_met=None
                    else:
                        ## hardcoding needs to correct.
                        print("super trend is not considered")
                        call_condition_met = eval(call_condition_str)               
                        put_condition_met=None

                    print(f" only call should be taken. call condition {call_condition_met} and put condition {put_condition_met}")
                    if call_condition_met:
                        print_str = "CALL CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                        print_str += f"Current Time: {datetime.datetime.now()}  \n"
                        print('CALL entry Conditions:',(call_condition_str),eval(call_condition_str))
                        print("Call conditions met and starting to take trade")
                                        
                        signal=True
                        if signal:
                            t1= cdf['date'].iloc[-1]
                            t2= cdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                listcheck.clear()
                                listcheck.append(t1)
                                listcheck.append(t2)
                                call_condition_met=True
                                put_condition_met=None
                                print("Since there is no existing put trades, going to take call trade")

                            else:
                                call_condition_met=None
                                put_condition_met=None
                                print("trade had been already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                               


                            
                    
                  
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df
                    
        
                elif BUY_ONLY_CALL_OR_PUT=="put":
                    if SUPERTREND=='yes':
                        previous_put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.                    
                        if previous_put_condition_met and supertrend_value=='down':
                            print("put condition met and super trend value is down so taking put trade")
                            put_condition_met=True
                            call_condition_met=None
                        else:
                            if previous_put_condition_met:
                                print("put condition met but super trend value is not down so taking not taking put trade")
                                put_condition_met=None
                                call_condition_met=None
                            else:
                                print("put condition was not satisfied in the first place")
                                put_condition_met=None
                                call_condition_met=None

                    else:
                        print("super trend is not considered")
                        put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.
                        call_condition_met = None

                    print(f" only put should be taken. call condition {call_condition_met} and put condition {put_condition_met}")
                    if put_condition_met:
                        print_str = "PUT CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                        print_str += f"Current Time: {datetime.datetime.now()}  \n"
                        print('PUT entry Conditions:',(put_condition_str),eval(put_condition_str))
                        print("Put conditions met and starting to take trade")
                        
                                            
                        signal=True
                        if signal:
                            t1= pdf['date'].iloc[-1]
                            t2= pdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                listcheck.clear()
                                listcheck.append(t1)
                                listcheck.append(t2)
                                call_condition_met=None
                                put_condition_met=True
                                print("Since there is no existing call trades, going to take put trade")
                              
                            else:
                                call_condition_met=None
                                put_condition_met=None 
                                print("trade had been already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                
                  
                    print(call_condition_met,put_condition_met)
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df


            elif INDEX_CHART== 'option_chart':
                if BUY_ONLY_CALL_OR_PUT=="both":     
                    #time.sleep(60)
                    if SUPERTREND=='yes':
                        # previous_put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.
                        previous_put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.
                        previous_call_condition_met = eval(call_condition_str)
                        # previous_call_condition_met = eval(call_condition_str)
                        print("call st value is",call_supertrend_value)
                        print("put st value is",put_supertrend_value)
                        if previous_call_condition_met and call_supertrend_value=='up':
                            print("call condition met and super trend value is up so taking call trade")
                            call_condition_met=True
                            put_condition_met=None
                        elif previous_put_condition_met and put_supertrend_value=='up':
                            print("put condition met and super trend value is down so taking put trade")
                            put_condition_met=True
                            call_condition_met=None
                        else:
                            if previous_call_condition_met or previous_put_condition_met:
                                print("call or put conditions met but super trend conditions havent been met ")
                                put_condition_met=None
                                call_condition_met=None
                            else:
                                print("call and put condition was not satisfied in the first place")
                                put_condition_met=None
                                call_condition_met=None

                    else:
                        print("supertrend is not considered")
                        put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.
                        call_condition_met = eval(call_condition_str)
                        # print('call df', cdf)
                    # print('put df', pdf)
                    # exit() ## hardcoding needs to correct.
                    
                    print(f" call condition {call_condition_met} and put condition {put_condition_met}")
                    if call_condition_met:
                        print_str = "CALL CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                        print_str += f"Current Time: {datetime.datetime.now()}  \n"
                        print('CALL entry Conditions:',(call_condition_str),eval(call_condition_str))
                        print("Call conditions met and starting to take trade")
                        
                        signal=True
                        if signal:
                            t1= cdf['date'].iloc[-1]
                            t2= cdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                print('t1 and t2 not in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                                

                                listcheck.clear()
                                listcheck.append(t1)
                                listcheck.append(t2)
                                
                                call_condition_met=True
                                put_condition_met=None
                            else:
                                print("trade had benn already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                                call_condition_met=None
                                put_condition_met=None

                                



                        

                    elif put_condition_met:
                        print_str = "PUT CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                        print_str += f"Current Time: {datetime.datetime.now()}  \n"                    
                        print('PUT entry Conditions:',(put_condition_str),eval(put_condition_str))
                        print("Put conditions met and starting to take trade")
                                        
                        signal=True
                        if signal:
                            t1= pdf['date'].iloc[-1]
                            t2= pdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                listcheck.clear()
                                listcheck.append(t1)
                                listcheck.append(t2)
                                
                                call_condition_met=None
                                put_condition_met=True
                                print("Since there is no existing call trades, going to take put trade")
                                
                            else:
                                call_condition_met=None
                                put_condition_met=None 
                                print("trade had benn already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                            
                    
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df
                    
            
                elif BUY_ONLY_CALL_OR_PUT=="call":
                    if SUPERTREND=='yes':
                        print("call st value is",call_supertrend_value)
                        print("put st value is",put_supertrend_value)
                        
                        previous_call_condition_met = eval(call_condition_str)
                        if previous_call_condition_met and call_supertrend_value=='up':
                            print("call condition met and super trend value is up so taking call trade")
                            call_condition_met=True
                            put_condition_met=None
                        
                        else:
                            if previous_call_condition_met:
                                print("call condition met but super trend value is not up so not  taking call trade")

                                put_condition_met=None
                                call_condition_met=None
                            else:
                                print("call condition was not satisfied in the first place")
                                put_condition_met=None
                                call_condition_met=None
                    else:
                        ## hardcoding needs to correct.
                        print("super trend is not considered")
                        call_condition_met = eval(call_condition_str)               
                        put_condition_met=None

                    print(f" only call should be taken. call condition {call_condition_met} and put condition {put_condition_met}")
                    if call_condition_met:
                        print_str = "CALL CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                        print_str += f"Current Time: {datetime.datetime.now()}  \n"
                        print('CALL entry Conditions:',(call_condition_str),eval(call_condition_str))
                        print("Call conditions met and starting to take trade")
                                        
                        signal=True
                        if signal:
                            t1= cdf['date'].iloc[-1]
                            t2= cdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                listcheck.clear()
                                listcheck.append(t1)
                                listcheck.append(t2)
                                call_condition_met=True
                                put_condition_met=None
                                print("Since there is no existing put trades, going to take call trade")
                                
                            else:
                                call_condition_met=None
                                put_condition_met=None
                                print("trade had been already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                                


                            
                    
                    
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df
                    
        
                elif BUY_ONLY_CALL_OR_PUT=="put":
                    print("call st value is",call_supertrend_value)
                    print("put st value is",put_supertrend_value)
                    if SUPERTREND=='yes':
                        previous_put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.                    
                        if previous_put_condition_met and put_supertrend_value=='up':
                            print("put condition met and super trend value is down so taking put trade")
                            put_condition_met=True
                            call_condition_met=None
                        else:
                            if previous_put_condition_met:
                                print("put condition met but super trend value is not down so taking not taking put trade")
                                put_condition_met=None
                                call_condition_met=None
                            else:
                                print("put condition was not satisfied in the first place")
                                put_condition_met=None
                                call_condition_met=None

                    else:
                        print("super trend is not considered")
                        put_condition_met = eval(put_condition_str) ## hardcoding needs to correct.
                        call_condition_met = None

                    print(f" only put should be taken. call condition {call_condition_met} and put condition {put_condition_met}")
                    if put_condition_met:
                        print_str = "PUT CONDITION SATISFIED" #f"Thread {threading.current_thread()}  \n"
                        print_str += f"Current Time: {datetime.datetime.now()}  \n"
                        print('PUT entry Conditions:',(put_condition_str),eval(put_condition_str))
                        print("Put conditions met and starting to take trade")
                        
                                            
                        signal=True
                        if signal:
                            t1= pdf['date'].iloc[-1]
                            t2= pdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                listcheck.clear()
                                listcheck.append(t1)
                                listcheck.append(t2)
                                call_condition_met=None
                                put_condition_met=True
                                print("Since there is no existing call trades, going to take put trade")
                                
                            else:
                                call_condition_met=None
                                put_condition_met=None 
                                print("trade had been already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                
                    
                    print(call_condition_met,put_condition_met)
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df



def full_logic(i,logFile,index_future,pe_tradingsymbol,ce_tradingsymbol,premium,time_frame_dict,listcheck):
            global bnf_tf1,opt_tf1
            user_opt_tf=call_time_frame
            user_bnf_tf=bnf_time_frame
            bnf_tf= time_frame_dict[user_bnf_tf]
            opt_tf= time_frame_dict[user_opt_tf]
            print('bnf_tf:',bnf_tf,'opt_tf:',opt_tf)
            bnf_tf1=bnf_tf;opt_tf1=opt_tf
    
            selected_values=(individual_values[f'candle_condition_{i}'], individual_values[f'previous_candle_{i}'],individual_values[f'close_{i}'],
                    individual_values[f'condition_{i}'],individual_values[f'call_candle_condition_{i}'],
                    individual_values[f'call_previous_candle_{i}'],individual_values[f'call_close_{i}'],
                    individual_values[f'call_condition_{i}'],individual_values[f'put_candle_condition_{i}'],
                    individual_values[f'put_previous_candle_{i}'],individual_values[f'put_close_{i}'],
                    individual_values[f'put_condition_{i}'], index_future,pe_tradingsymbol,ce_tradingsymbol)##last strike price from premium pending
            
            
           
            
            call_trade,put_trade,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df = create_trade_conditions(*selected_values,logFile, i, bnf_tf,opt_tf,listcheck)
            print('new',call_trade,put_trade,supertrend_value)
           
            
            if call_trade:
                        print("Given conditions:",selected_values)
                        
                        trade="call"
                        name= ce_tradingsymbol
                        ltp= get_ltp(name,logFile)
                        print("ltp is", ltp)
                        print("permissible lower limit is", premium_limit)
                        if  premium_limit > ltp:
                            allowed=premium_limit
                            print(f'ltp {ltp} is lesser than permissible premium range {allowed} so not proceeding with trade. Scanning starting again')

                            call_trade=False
                        else:
                            allowed=premium_limit
                            print(f'ltp {ltp} is greater than permissible premium range {allowed} so proceeding with trade')

                        #opt_tf=individual_values[f'call_time_frame_{i}']
            if put_trade:
                        print("Given conditions:",selected_values)
                        
                        trade="put"
                        name= pe_tradingsymbol
                        ltp= get_ltp(name)
                        print("ltp is", ltp)
                        print("permissible lower limit is", premium_limit)
                        if  premium_limit > ltp:
                            allowed=individual_values[f'premium_limit_{i}']
                            print(f'ltp {ltp} is lesser than permissible premium range {allowed} so not proceeding with trade. Scanning starting again')

                            put_trade=False
                        else:
                            allowed=premium_limit
                            print(f'ltp {ltp} is greater than permissible premium range {allowed} ')


                        #opt_tf=individual_values[f'put_time_frame_{i}']
                          
          
            trade_checking={}
            call_trade_taken=False
            put_trade_taken=False
            print("order type is",ORDER_TYPE)


            ###ORDER TYPE LOGICS####
            if call_trade:             
              trade_ltp,co= get_ltp_and_current_opt_open(name,logFile,opt_tf)
              quantity= get_quantity(trade_ltp,i,logFile)
              
              qty=quantity
              local_quantity=quantity
              if quantity != 0:
                print("Quantity is",local_quantity)            
                print("checking trade:",name,'BUY',quantity,type(quantity))
                logFile.flush(); os.fsync(logFile.fileno())
                if individual_values[f'order_type_{i}']=='market':
                    order_type= 'market'
                    carry= individual_values[f'product_type_{i}']
                    limit_value=0
                    order_id = orderplacement.order_placement(name,'BUY',quantity,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                    if status=='COMPLETE':              
                        call_trade_taken= True
                    else:
                        print("error in placing order checking conditions again")
                        call_trade=False
                    #order_exec_price= orderplacement.get_averageprice(name)
                    
                    print("order exec price", order_exec_price)           
                    target_time= time_calc(i,opt_tf,entry=False)
                    print('target time is', target_time) ##logic for taking order and getting to know order execution price

                elif individual_values[f'order_type_{i}']=='limit':
                    order_type='limit'
                    carry= individual_values[f'product_type_{i}']

                    print("Type of limit value chosen is", individual_values[f'limit_order_value_{i}'])
                    if individual_values[f'limit_order_value_{i}']=='open':
                        limit_value= int(CO2)
                    elif individual_values[f'limit_order_value_{i}']=='high':
                        limit_value= int(CH2)
                    elif individual_values[f'limit_order_value_{i}']=='low':
                        limit_value= int(CL2)
                    elif individual_values[f'limit_order_value_{i}']=='close':
                        limit_value= int(CC2)
                    order_id = orderplacement.order_placement(name,'BUY',quantity,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                    print("The set limit order value is",limit_value)                     
                    target_time=time_calc(i,opt_tf,entry=True)
                    print("target time is",target_time)
                    nowtimee = datetime.datetime.now()
                    while nowtimee<target_time:
                        order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                        if status=='COMPLETE':
                            call_trade_taken= True
                            target_time=time_calc(i,opt_tf,entry=False)
                            print("target time for exit at mkt inside limit order",target_time)
                            break
                        elif status=="OPEN":
                            call_trade_taken= False
                            put_trade_taken= False
                            print("limit order value is", limit_value)
                            print("limit order status is still open and pending, checking again") 
                        nowtimee = datetime.datetime.now()
                    if (call_trade_taken== False and  put_trade_taken==False):
                        print("Target time is reached going to scan again")
                    if status=="OPEN":
                        #cancel_order = kite.cancel_order(order_id = placed_order_id, variety='regular', parent_order_id = None)
                        try:
                            response = kite.cancel_order(order_id=order_id,variety='regular')
                            
                            print("Order cancelled successfully", response)
                        except Exception as e:
                            print("Error in cancelling order:", e) 

                        
                elif individual_values[f'order_type_{i}']=='slm':
                    order_type='slm'
                    carry= individual_values[f'product_type_{i}']
                    print("carry is ",carry)                    
                    print("Type of limit value chosen is", individual_values[f'limit_order_value_{i}'])
                    if individual_values[f'limit_order_value_{i}']=='open':
                        limit_value= float(CO2)
                    elif individual_values[f'limit_order_value_{i}']=='high':
                        limit_value= float(CH2)
                    elif individual_values[f'limit_order_value_{i}']=='low':
                        limit_value= float(CL2)
                    elif individual_values[f'limit_order_value_{i}']=='close':
                        limit_value= float(CC2)

                    ltp = get_ltp(name,logFile)
                    print("ltp is", ltp)
                    print("limit_value is", limit_value)
                    if limit_value>ltp:
                                       
                        order_id = orderplacement.order_placement(name,'BUY',quantity,logFile,kite,Order_Identifier,order_type,carry,limit_value)  
                        order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                        print("status is",status)

                    else:
                        print("ltp is greater than trigger price")
                        status=None

                    if status=='OPEN':
                        print("the status is OPEN,Going to cancel the trade")
                        try:
                            response = kite.cancel_order(order_id=order_id,variety='regular')                            
                            print("Order cancelled successfully", response)
                        except Exception as e:
                            print("Error in cancelling order:", e) 

                        status=None

                    if status!="TRIGGER PENDING":                       
                        call_trade_taken=False
                        put_trade_taken=False
                    else:
                        print("slm order has been placed")
                        order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 

                        print("The set limit order value is",limit_value)                     
                        new_logic_slm=False
                        while not new_logic_slm:
                            target_time=time_calc(i,opt_tf,entry=True)
                            print("target time is",target_time)
                            nowtimee = datetime.datetime.now()
                            while nowtimee<target_time:
                                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                                if status=='COMPLETE':
                                    call_trade_taken= True
                                    new_logic_slm=True
                                    filename="open_position_tracker.txt"
                                    append_to_file(filename, "call",logFile)
                                    target_time=time_calc(i,opt_tf,entry=False)
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
                                    response = kite.cancel_order(order_id=order_id,variety='regular')
                                    
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
              trade_ltp,co= get_ltp_and_current_opt_open(name,logFile,opt_tf)
              quantity= get_quantity(trade_ltp,i,logFile)
              qty=quantity
              local_quantity=quantity
              if quantity != 0:
                if individual_values[f'order_type_{i}']=='market':
                    order_type= 'market'
                    carry= individual_values[f'product_type_{i}']
                    limit_value=0
                    print("Quantity is",local_quantity)
                    order_id = orderplacement.order_placement(name,'BUY',quantity,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                    if status=='COMPLETE':              
                        put_trade_taken= True
                    else:
                        print("error in placing order checking conditions again")
                        put_trade=False
                    
                    target_time= time_calc(i,opt_tf,entry=False)
                    print("target time is", target_time)


                elif individual_values[f'order_type_{i}']=='limit':
                    order_type='limit'
                    carry= individual_values[f'product_type_{i}']

                    print("Type of limit value chosen is", individual_values[f'limit_order_value_{i}'])
                    if individual_values[f'limit_order_value_{i}']=='open':
                        limit_value= int(PO2)
                    elif individual_values[f'limit_order_value_{i}']=='high':
                        limit_value= int(PH2)
                    elif individual_values[f'limit_order_value_{i}']=='low':
                        limit_value= int(PL2)
                    elif individual_values[f'limit_order_value_{i}']=='close':
                        limit_value= int(PC2)

                    print("The set limit order value is",limit_value)                     

                    order_id = orderplacement.order_placement(name,'BUY',quantity,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                                            
                    target_time=time_calc(i,opt_tf,entry=True)
                    print("target time is",target_time)
                    nowtimee = datetime.datetime.now()
                    while nowtimee<target_time:
                        order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                        if status=='COMPLETE':
                            put_trade_taken= True
                            target_time=time_calc(i,opt_tf,entry=False)
                            print("target time for exit at mkt inside limit order",target_time)
                            break
                        elif status=="OPEN":
                            call_trade_taken= False
                            put_trade_taken= False
                            print("limit order value is", limit_value)
                            print("limit order status is still open and pending, checking again") 
                        nowtimee = datetime.datetime.now()
                    if status=="OPEN":
                        #cancel_order = kite.cancel_order(order_id = placed_order_id, variety='regular', parent_order_id = None)
                        try:
                            response = kite.cancel_order(order_id=order_id,variety='regular')
                            print("Order cancelled successfully", response)
                        except Exception as e:
                            print("Error in cancelling order:", e)

                elif individual_values[f'order_type_{i}']=='slm':
                    order_type='slm'
                    carry= individual_values[f'product_type_{i}']
                    print("carry is ",carry)
                    print("Type of limit value chosen is", individual_values[f'limit_order_value_{i}'])
                    if individual_values[f'limit_order_value_{i}']=='open':
                        limit_value= float(PO2)
                    elif individual_values[f'limit_order_value_{i}']=='high':
                        limit_value= float(PH2)
                    elif individual_values[f'limit_order_value_{i}']=='low':
                        limit_value= float(PL2)
                    elif individual_values[f'limit_order_value_{i}']=='close':
                        limit_value= float(PC2)

                    ltp = get_ltp(name,logFile)
                    print("ltp is", ltp)
                    print("limit_value is", limit_value)
                    if limit_value>ltp:
                        print("The set limit order value is",limit_value) 
                        order_id = orderplacement.order_placement(name,'BUY',quantity,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                        order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                        print("status is",status)
                    else:
                        print("ltp is greater than trigger price")
                        status=None


                    if status=='OPEN':
                        print("the status is OPEN,Going to cancel the trade")
                        try:
                            response = kite.cancel_order(order_id=order_id,variety='regular')                            
                            print("Order cancelled successfully", response)
                        except Exception as e:
                            print("Error in cancelling order:", e) 

                        status=None

                    if status!="TRIGGER PENDING":                       
                        call_trade_taken=False
                        put_trade_taken=False
                    
                    else:
                        order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                        new_logic_slm=False
                        while not new_logic_slm:                        
                            target_time=time_calc(i,opt_tf,entry=True)
                            print("target time is",target_time)
                            nowtimee = datetime.datetime.now()
                            while nowtimee<target_time:
                                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',order_id) 
                                if status=='COMPLETE':
                                    put_trade_taken= True
                                    new_logic_slm=True
                                    filename="open_position_tracker.txt"
                                    append_to_file(filename, "put",logFile)
                                    target_time=time_calc(i,opt_tf,entry=False)
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
                                    response = kite.cancel_order(order_id=order_id,variety='regular')
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
                                 
                    if (individual_values[f'isl_{i}']== 'Call Option Candle Lowest Between P-2 and P-1 candle low' and not all_exited) :                       
                        
                        if call_trade_taken:
                            sl=get_isl(i,cdf,logFile,order_exec_price) 
                            # sl=10                                             
                                                  
                        if put_trade_taken:
                            sl=get_isl(i,pdf,logFile,order_exec_price)                    
                        
    
                        sl_price= round(float(sl),1)
                        isl_value=sl_price
                        print("isl is",sl_price)
                        logFile.flush(); os.fsync(logFile.fileno())
    
                        price= round(float(sl-0.1),1)
                        ltp= get_ltp(name,logFile)
                        if sl_price>ltp:
                            print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                            logFile.flush(); os.fsync(logFile.fileno())
                            carry= individual_values[f'product_type_{i}']
                            limit_value=0                               
                            oid=orderplacement.order_placement(name,'SELL',qty,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                            
                            order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                            if status=="COMPLETE":
                                all_exited=True
                                
                           
                        else:
                            carry= individual_values[f'product_type_{i}']
                            if carry=='normal':
                                product_type=kite.PRODUCT_NRML
                            elif carry=='intraday':
                                product_type=kite.PRODUCT_MIS
                                    
                            sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(carry,all_exited,name,qty,kite.ORDER_TYPE_SL,product_type,price,kite.TRANSACTION_TYPE_SELL,kite.EXCHANGE_NFO,kite.VARIETY_REGULAR,sl_price,kite.VALIDITY_DAY,logFile,kite,Order_Identifier)        


                            #sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(all_exited,name,quantity,kite.ORDER_TYPE_SL,kite.PRODUCT_MIS,price,kite.TRANSACTION_TYPE_SELL,kite.EXCHANGE_NFO,kite.VARIETY_REGULAR,sl_price,kite.VALIDITY_DAY,logFile,kite,Order_Identifier)                    
                            

                            print("sl oredr id", sl_order_id)
                            print("placed isl")
                            logFile.flush(); os.fsync(logFile.fileno())
                        c=0;o=0;rsi=0;opt_open=0;opt_close=0;orsi=0                    
                        isl=True
                        
    
                    elif (individual_values[f'isl_{i}']== 'Call Option Last Bearish Candle Low' and not all_exited):
                        
                        if call_trade_taken:
                            sl=get_last_bearish_candle(i,cdf,logFile,order_exec_price)
                                               
                        if put_trade_taken:
                            sl=get_last_bearish_candle(i,pdf,logFile,order_exec_price)
                        
                        sl_price= round(float(sl),1)
                        isl_value= sl_price
                        print("isl is",sl_price)
    
                        price= round(float(sl-0.1),1) 
                        ltp= get_ltp(name,logFile)
                        if sl_price>ltp:
                            print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                            logFile.flush(); os.fsync(logFile.fileno())
                            carry= individual_values[f'product_type_{i}']
                            limit_value=0                              
                            oid=orderplacement.order_placement(name,'SELL',qty,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                    
                            #oid=orderplacement.order_placement(name,'SELL',quantity,logFile,kite,Order_Identifier)
                            order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                            if status=="COMPLETE":
                                all_exited=True
                                
                        else:

                            carry= individual_values[f'product_type_{i}']
                            if carry=='normal':
                                product_type=kite.PRODUCT_NRML
                            elif carry=='intraday':
                                product_type=kite.PRODUCT_MIS
                            sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(carry,all_exited,name,qty,kite.ORDER_TYPE_SL,product_type,price,kite.TRANSACTION_TYPE_SELL,kite.EXCHANGE_NFO,kite.VARIETY_REGULAR,sl_price,kite.VALIDITY_DAY,logFile,kite,Order_Identifier)        
                     
                            

                            
                            print("sl oredr id", sl_order_id)
                            print("placed isl")
                            logFile.flush(); os.fsync(logFile.fileno())
                        
                        c=0;o=0;rsi=0;opt_open=0;opt_close=0;orsi=0                   
                        logFile.flush(); os.fsync(logFile.fileno())
                
                        isl=True

                    elif (individual_values[f'isl_{i}']== 'Not applicable' and not all_exited):
                        sl=(order_exec_price - (individual_values[f'isl_percentage_{i}'] *order_exec_price))
                        sl_price= round(float(sl),1)
                        isl_value= sl_price
                        print("isl is",sl_price)
                        price= round(float(sl-0.1),1)
                       
                        ltp= get_ltp(name,logFile)
                        if sl_price>ltp:
                            print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                            logFile.flush(); os.fsync(logFile.fileno())
                            carry= individual_values[f'product_type_{i}']
                            limit_value=0                                
                            oid=orderplacement.order_placement(name,'SELL',qty,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                    
                            order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                            if status=="COMPLETE":
                                print("exited")
                        else:                                    
                            carry= individual_values[f'product_type_{i}']
                            if carry=='normal':
                                product_type=kite.PRODUCT_NRML
                            elif carry=='intraday':
                                product_type=kite.PRODUCT_MIS
                            sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(carry,all_exited,name,qty,kite.ORDER_TYPE_SL,product_type,price,kite.TRANSACTION_TYPE_SELL,kite.EXCHANGE_NFO,kite.VARIETY_REGULAR,sl_price,kite.VALIDITY_DAY,logFile,kite,Order_Identifier)        
                     
                            if all_exited:
                                print("exited the trade")

                            
                            print("sl oredr id", sl_order_id)
                            print("placed isl")
                            logFile.flush(); os.fsync(logFile.fileno())
                        
                        c=0;o=0;rsi=0;opt_open=0;opt_close=0;orsi=0                   
                        logFile.flush(); os.fsync(logFile.fileno())
                
                        isl=True
    
                    elif (individual_values[f'isl_{i}']== 'Bearish candle below LL' and not all_exited):
                        if call_trade_taken:
                            sl=get_last_bearish_candle_with_rsi(i,ce_rsi_df,logFile,order_exec_price,individual_values[f'lower_limit_{i}'])
                                               
                        if put_trade_taken:
                            sl=get_last_bearish_candle_with_rsi(i,pe_rsi_df,logFile,order_exec_price,individual_values[f'lower_limit_{i}'])
                        
                        sl_price= round(float(sl),1)
                        isl_value= sl_price
                        print("isl is",sl_price)
    
                        price= round(float(sl-0.1),1) 
                        ltp= get_ltp(name,logFile)
                        if sl_price>ltp:
                            print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                            logFile.flush(); os.fsync(logFile.fileno())
                            carry= individual_values[f'product_type_{i}']
                            limit_value=0                              
                            oid=orderplacement.order_placement(name,'SELL',qty,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                    
                            #oid=orderplacement.order_placement(name,'SELL',quantity,logFile,kite,Order_Identifier)
                            order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_exec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                            if status=="COMPLETE":
                                all_exited=True
                                
                        else:

                            carry= individual_values[f'product_type_{i}']
                            if carry=='normal':
                                product_type=kite.PRODUCT_NRML
                            elif carry=='intraday':
                                product_type=kite.PRODUCT_MIS
                            sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(carry,all_exited,name,qty,kite.ORDER_TYPE_SL,product_type,price,kite.TRANSACTION_TYPE_SELL,kite.EXCHANGE_NFO,kite.VARIETY_REGULAR,sl_price,kite.VALIDITY_DAY,logFile,kite,Order_Identifier)        
                     
                            

                            
                            print("sl oredr id", sl_order_id)
                            print("placed isl")
                            logFile.flush(); os.fsync(logFile.fileno())
                        
                        c=0;o=0;rsi=0;opt_open=0;opt_close=0;orsi=0                   
                        logFile.flush(); os.fsync(logFile.fileno())
                
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
                        order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',sl_order_id)
                        print("status of SL order", status)
                        logFile.flush(); os.fsync(logFile.fileno())
    
                        if status=='COMPLETE':
                            all_exited=True
                            if target_order_id !=None:
                                try:
                                    response = kite.cancel_order(order_id=target_order_id,variety='regular')                            
                                    print("Order cancelled successfully", response)
                                except Exception as e:
                                    print("Error in cancelling order:", e)


                        elif status=='OPEN':
                            print("status got changed to open so exiting all")
                            oid= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
                            all_exited= True
                            local_quantity=0


                        current_time= datetime.datetime.now().time()
                        exit_time= individual_values[f'exit_time_{i}']                       
                        if current_time>=exit_time:
                            oid= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
                            time.sleep(5)
                            #status= orderplacement.get_orderstatus(oid,logFile)
                            order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_xec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                            if status=='COMPLETE':
                                all_exited=True
                                local_quantity=0
                                if target_order_id !=None:
                                    try:
                                        response = kite.cancel_order(order_id=target_order_id,variety='regular')                            
                                        print("Order cancelled successfully", response)
                                    except Exception as e:
                                        print("Error in cancelling order:", e)
                                
                                print("Time is greater than exit time exiting all")
                                                                                                                                           
                        else:
                            
                            now= datetime.datetime.now()
                            if 4 < now.second < 50 and not all_exited:
                                rsi_length= individual_values[f'rsi_{i}']
                                new_close, new_open, new_rsi, prev_rsi,new_low,last_candle_time = get_close_and_rsi(index_future,bnf_tf,4,'NFO',logFile,rsi_length )
                                new_opt_close, new_opt_open, new_orsi, prev_orsi,new_opt_low,last_candle_time= get_close_and_rsi(name,opt_tf,4,'NFO',logFile,rsi_length)
                                Cll1,Cll2,clO2,clH2,clL2,clRSI,nadf,cprev_rsi,ccandle_p1_time,rsi_rsi= get_data(index_future,bnf_tf,4,'NFO',logFile,rsi_length)
                                ts1_condition_satisfied=False ; ts2_condition_satisfied=False
    
                                print('new_close',new_close, 'new_open', new_open, 'new_rsi',new_rsi,'new_opt_close',new_opt_close, 'new_opt_open', new_opt_open, 'new_orsi',new_orsi )                        
                                logFile.flush(); os.fsync(logFile.fileno())
                                
                                if (call_trade_taken and (prev_orsi>individual_values[f'upper_limit_{i}'])):
                                     ts1_condition_satisfied= True
                                     print('CC','prev_rsi:',prev_rsi, 'prev_orsi:',prev_orsi,'UL:',individual_values[f'upper_limit_{i}'])
                                
                                if (call_trade_taken  and (prev_orsi>individual_values[f'lower_limit_{i}'])):
                                     ts2_condition_satisfied= True
                                     print('CC','prev_rsi:',prev_rsi, 'prev_orsi:',prev_orsi,'UL:',individual_values[f'upper_limit_{i}'])
                                
                                if (put_trade_taken  and (prev_orsi>individual_values[f'upper_limit_{i}'])):
                                     ts1_condition_satisfied= True
                                     print('PC','prev_rsi:',prev_rsi, 'prev_orsi:',prev_orsi,'UL:',individual_values[f'upper_limit_{i}'], 'LL:', individual_values[f'lower_limit_{i}'])
    
                                if (put_trade_taken and (prev_orsi>individual_values[f'lower_limit_{i}'])):
                                     ts2_condition_satisfied= True
                                     print('PC','prev_rsi:',prev_rsi, 'prev_orsi:',prev_orsi,'UL:',individual_values[f'upper_limit_{i}'], 'LL:', individual_values[f'lower_limit_{i}'])
    
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
                                    print('\nCondition checking!:',print_str,'UL-->:',individual_values[f'upper_limit_{i}'],'LL-->:',individual_values[f'lower_limit_{i}'])                                   
                                    print('last candle time', last_candle_time)
                                    st=get_supertrend(nadf,logFile,i)
                                    print("Future super trend value is",st)
                                    logFile.flush(); os.fsync(logFile.fileno())                               
                                    entered_condition = False
                                    bullish_rsi=True

                        
                                print_str = f"Current Time: {datetime.datetime.now()}  \n"
                                print_str += f"1. {name} P1 candle close: {opt_close} | 2. {name} P1 candle open: {opt_open} | 3. {name} P1 candle RSI: {orsi} | 4. {name} P1 candle low: {opt_low} | 5. P1 {name} candle time: {last_candle_time} | 6. {name} P2 RSI:{prev_orsi} \n"
                                print_str += f"7. {index_future} P1 candle close: {c} | 8. {index_future} P1 candle open: {o} | 9. {index_future} P1 candle RSI: {rsi} | 10. {index_future} P1 candle low: {fut_low} | 11. P1 {index_future} candle time: {last_candle_time} | 12. {index_future} P2 RSI:{prev_rsi} \n"
                                print('\nCondition checking!:',print_str,'UL-->:',individual_values[f'upper_limit_{i}'],'LL-->:',individual_values[f'lower_limit_{i}'])
                                


                                #### 1A/1B AND TARGET LOGICS #####
                                if ((individual_values[f'tsl1_{i}'][3:] == 'Exit at market price' or individual_values[f'tsl2_{i}'][3:] == 'Exit at market price')  and not all_exited):                       
                                    now = datetime.datetime.now()
                                    if now> target_time and not all_exited:
                                        if orsi > individual_values[f'bullish_rsi_limit_{i}'] and bullish_rsi and individual_values[f'bullish_rsi_enabler_{i}']=="enable":
                                            if opt_low > order_exec_price:
                                                bullish_rsi=False
                                                print(f"p1 rsi: {orsi} is greater than the given rsi value:{individual_values[f'bullish_rsi_limit_{i}']}")
                                                sl_price= round(float(opt_low),1)
                                                price= round(float(opt_low-0.1),1) 
                                                sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                                                isl_value=opt_low
                                                print("Shifting rsi based tsl value")
                                                print("New tsl is",isl_value)
                                                logFile.flush(); os.fsync(logFile.fileno())
                                            else:
                                                print("option low is not greater than order executed price so not shifting rsi based sl")
                                    
                                        elif opt_close < opt_open:                                    
                                            all_exited,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,local_quantity,ssl=checking_condition(trade,ssl,opt_low,order_exec_price,i,name,opt_tf,logFile,sl_order_id,all_exited,local_quantity,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                        elif opt_close > opt_open and individual_values[f'bullish_condition_enabler_{i}']=="enable":
                                            print("P1 candle is bullish so checking the p2 and p3 candles")
                                            sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,i,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,4,'NFO',logFile,rsi_length)
                                            
                                    else:                        
                                        if not all_exited:
                                            sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                            ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)
                                            o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,logFile,opt_tf)
                                            all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)
                        
                                #### 2A/2B AND TARGET LOGICS #####
                                elif ((individual_values[f'tsl1_{i}'][3:]== 'Full trailing with bearish candle low sl') or (individual_values[f'tsl2_{i}'][3:]== 'Full trailing with bearish candle low sl'))and not all_exited:
                                    
                                    if orsi > individual_values[f'bullish_rsi_limit_{i}'] and bullish_rsi and individual_values[f'bullish_rsi_enabler_{i}']=="enable":
                                        if opt_low > order_exec_price:
                                            
                                            bullish_rsi=False
                                            print(f"p1 rsi: {orsi} is greater than the given rsi value:{individual_values[f'bullish_rsi_limit_{i}']}")
                                            sl_price= round(float(opt_low),1)
                                            price= round(float(opt_low-0.1),1) 
                                            sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                                            
                                            isl_value=opt_low
                                            print("Shifting rsi based tsl value")
                                            print("New tsl is",isl_value)
                                            logFile.flush(); os.fsync(logFile.fileno())
                                        else:
                                            print("option low is not greater than order executed price so not shifting rsi based sl")
                                    
                                    
                                    elif opt_close < opt_open:
                                        print("p1 candle is bearish")
                                        logFile.flush(); os.fsync(logFile.fileno())

                                        sl_order_id,isl_value,all_exited= full_trailing_checking(trade,opt_low,isl_value,sl_order_id,logFile,all_exited,local_quantity)
                                    
                                    elif opt_close > opt_open and individual_values[f'bullish_condition_enabler_{i}']=="enable":
                                        print("P1 candle is bullish so checking the p2 and p3 candles")
                                        sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,i,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,4,'NFO',logFile,rsi_length)
                                                                        
                                    if not all_exited:
                                        sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                        o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,logFile,opt_tf)
                                        ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)
                                        
                                        all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)
        
                                #### 3A/3B AND TARGET LOGICS #####
                                elif ((individual_values[f'tsl1_{i}'][3:]== 'Exit half and remaining at bearish candle low') or (individual_values[f'tsl2_{i}'][3:]== 'Exit half and remaining at bearish candle low') or (individual_values[f'tsl2_{i}'][3:]== 'Exit half and remaining at bearish candle low above sl price'))  and not all_exited:
                                    
                                    if orsi > individual_values[f'bullish_rsi_limit_{i}'] and bullish_rsi and individual_values[f'bullish_rsi_enabler_{i}']=="enable":
                                        if opt_low > order_exec_price:
                                                                            
                                            bullish_rsi=False
                                            print(f"p1 rsi: {orsi} is greater than the given rsi value:{individual_values[f'bullish_rsi_limit_{i}']}")
                                            sl_price= round(float(opt_low),1)
                                            price= round(float(opt_low-0.1),1) 
                                            sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                                            

                                            isl_value=opt_low
                                            print("Shifting rsi based tsl value")
                                            print("New tsl is",isl_value)
                                            logFile.flush(); os.fsync(logFile.fileno())
                                        else:
                                            print("option low is not greater than order executed price so not shifting rsi based sl")
                                    
                                    
                                    elif opt_close < opt_open:
                                        
                                        if opt_low > order_exec_price:
                                            
                                            logFile.flush(); os.fsync(logFile.fileno())
                                            print("opt close is greater than order executed price and p1 candle is bearish")
                                            print("opt close is",opt_close,"order executed price is",order_exec_price)
                                            logFile.flush(); os.fsync(logFile.fileno())
                                            if individual_values[f'tsl1_{i}'][3:]== 'Exit half and remaining at bearish candle low':
                                                print(" 3A is chosen")

                                                all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                            elif individual_values[f'supertrend_trailing_{i}']=="enable" and individual_values[f'tsl2_{i}'][3:]== 'Exit half and remaining at bearish candle low':
                                                print("super trend based trailing is enabled and 3B is chosen")
                                                
                                                if trade=="call" and st=="down":
                                                    print("call trade is taken and super trend is red so applying 3B")

                                                    all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                                else:
                                                    print("call trade is taken and super trend is still green so not applying 3A/3B")
                                                if trade=="put" and st=="up":
                                                    print("put trade is taken and super trend is green so applying 3A")

                                                    all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                                else:
                                                    print("put trade is taken and super trend is still red so not applying 3A/3B")

                                            elif individual_values[f'supertrend_trailing_{i}']=="disable" and individual_values[f'tsl2_{i}'][3:]== 'Exit half and remaining at bearish candle low':
                                                print(" 3B is chosen and st is disable")
                                                all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)

                                        elif  opt_low > isl_value and individual_values[f'tsl2_{i}'][3:]== 'Exit half and remaining at bearish candle low above sl price':
                                            print("option low is",opt_low)
                                            print("isl_value is",isl_value)
                                            print("Option low is greater than isl so going to apply 3b")
                                            if trade=="call" and st=="down":
                                                    print("call trade is taken and super trend is red so applying 3B")

                                                    all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                            else:
                                                print("call trade is taken and super trend is still green so not applying 3A/3B")
                                            if trade=="put" and st=="up":
                                                print("put trade is taken and super trend is green so applying 3A")

                                                all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                            else:
                                                print("put trade is taken and super trend is still red so not applying 3A/3B")

                                            

                                            
                                                                                        
                                        else:
                                            print(" option close is not greater than order executed price so not executing tsl1")

                                    elif opt_close > opt_open and individual_values[f'bullish_condition_enabler_{i}']=="enable":
                                        print("P1 candle is bullish so checking the p2 and p3 candles")
                                        sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,i,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,4,'NFO',logFile,rsi_length)
                                    

                                    if not all_exited:
                                        sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                        o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,logFile,opt_tf)
                                        ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)

                                        all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)
        

                                #### 4A/4B AND TARGET LOGICS #####
                                elif ((individual_values[f'tsl1_{i}'][3:]== 'Exit at market price and half trailing with bearish candle low sl') or (individual_values[f'tsl2_{i}'][3:]== 'Exit at market price and half trailing with bearish candle low sl')) and not all_exited:
                                    now = datetime.datetime.now()
                                    if now> target_time and not all_exited:
                                        if orsi > individual_values[f'bullish_rsi_limit_{i}'] and bullish_rsi and individual_values[f'bullish_rsi_enabler_{i}']=="enable":
                                            
                                            if opt_low > order_exec_price:

                                                bullish_rsi=False
                                                print(f"p1 rsi: {orsi} is greater than the given rsi value:{individual_values[f'bullish_rsi_limit_{i}']}")
                                                sl_price= round(float(opt_low),1)
                                                price= round(float(opt_low-0.1),1) 
                                                sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                                                
                                                isl_value=opt_low
                                                print("Shifting rsi based tsl value")
                                                print("New tsl is",isl_value)
                                                logFile.flush(); os.fsync(logFile.fileno())
                                            else:
                                                print(" option close is not greater than order executed price so not executing tsl1")
                                        
                                        elif opt_close < opt_open:
                                            if opt_low < order_exec_price and not all_exited:
                                                exit_all= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
                                                #status=orderplacement.get_orderstatus(exit_all,logFile)
                                                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',exit_all)
                                                if status=='COMPLETE':
                                                    all_exited= True
                                                    
                                                    print("exited all")
                                                    logFile.flush(); os.fsync(logFile.fileno())                                    
                                                    local_quantity=0
                                            else:
                                                if not entered_condition:
                                                    
                                                    all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                                    

                                        elif opt_close > opt_open and individual_values[f'bullish_condition_enabler_{i}']=="enable":
                                            print("P1 candle is bullish so checking the p2 and p3 candles")
                                            sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,i,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,4,'NFO',logFile,rsi_length)
                                    
                                        
                                        if not all_exited:
                                                sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                                o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,logFile,opt_tf)
                                                ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)
                                                all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)
                                    else:
                                        if not all_exited:
                                                sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                                o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,logFile,opt_tf)
                                                ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)
                                                all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)
                
                                 #### 5A/5B AND TARGET LOGICS #####
                                
                                #### 5A/5B AND TARGET LOGICS #####
                                elif ((individual_values[f'tsl1_{i}'][3:]== 'Full trailing and half trailing with bearish candle low sl') or (individual_values[f'tsl2_{i}'][3:]== 'Full trailing and half trailing with bearish candle low sl')) and not all_exited:
                                    if orsi > individual_values[f'bullish_rsi_limit_{i}'] and bullish_rsi and individual_values[f'bullish_rsi_enabler_{i}']=="enable":
                                            
                                        if opt_low > order_exec_price:
                                            
                                            bullish_rsi=False
                                            print(f"p1 rsi: {orsi} is greater than the given rsi value:{individual_values[f'bullish_rsi_limit_{i}']}")
                                            sl_price= round(float(opt_low),1)
                                            price= round(float(opt_low-0.1),1) 
                                            sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                                            

                                            isl_value=opt_low
                                            print("Shifting rsi based tsl value")
                                            print("New tsl is",isl_value)
                                            logFile.flush(); os.fsync(logFile.fileno())
                                        else:
                                                print(" option close is not greater than order executed price so not executing tsl1")
                                    
                                    elif opt_close < opt_open:
                                        if opt_low < order_exec_price:
                                            sl_order_id,isl_value,all_exited= full_trailing_checking(trade,opt_low,isl_value,sl_order_id,logFile,all_exited,local_quantity)
                                            
                                        else:
                                            all_exited, sl_order_id,local_quantity,entered_condition,isl_value= check_tsl_with_extra_condition(prevrsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity)
                                    
                                    elif opt_close > opt_open and individual_values[f'bullish_condition_enabler_{i}']=="enable":
                                        print("P1 candle is bullish so checking the p2 and p3 candles")
                                        sl_order_id,isl_value,all_exited,bull_condt= bullish_condition(bull_condt,i,order_exec_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name,opt_tf,4,'NFO',logFile,rsi_length)
                                    
                                    
                                    if not all_exited:
                                            sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                            o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,logFile,opt_tf)
                                            ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)
                                            all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)
            
                            else:
                                if not all_exited:
                                    sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
                                    o_ltp,curent_candle_open=get_ltp_and_current_opt_open(name,logFile,opt_tf)
                                    ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)
                                    all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)

                    t_time= time_calc(i,opt_tf,True)
                    now = datetime.datetime.now()
                    while now<t_time:
                        
                        now = datetime.datetime.now()

                        
                    print("Going to take trade")


            return listcheck



##########################################################

if __name__ == '__main__':
    # Example usage of get_historic_data
    #data = get_historic_data("BTCUSDT", "1h")
    
    # Example usage of Heikin-Ashi calculations
    #ha_data = heikin_ashi(data)
    #print(ha_data)

    info = get_qty("BTCUSDT")
    print(info)
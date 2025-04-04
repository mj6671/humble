#import pandas as pd
import threading
import datetime
import talib
import time
import math
import sys
import os
import os.path
import orderplacement
import Strike_Selection
import warnings
import csv
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')



def Nice_day_BEGIN(strategy_no1,kite1,premium_bucket,pd,folder):
  pd.options.mode.chained_assignment = None 
  token_dict = {}
  #print('Inititing strategy',strategy_no1)
  global strategy_no,kite
  #lock = threading.Lock()
  strategy_no = strategy_no1;kite = kite1
  premium_bucket-=1
  if True:
    df = pd.read_excel('Index_BUY_Algo_Configurations.xlsm')
    df[df.columns[0]] = df[df.columns[0]].str.lower().replace(' ', '_', regex=True)
    data = {}
    
    for i, row in df.iterrows():
        if i < 1: 
            continue
        key = row[df.columns[0]]
        data[key] = list(row[1:])
        
    individual_values = {}
    
    for key in data:
        for i, val in enumerate(data[key], start=1):
            individual_values[f'{key}_{i}'] = val
    
    #semaphore = threading.Semaphore(5)
    
    time_frame_dict = {
        '1 min': 'minute',
        '2 mins': '2minute',
        '3 mins': '3minute',
        '4 mins': '4minute',
        '5 mins': '5minute',
        '10 mins': '10minute',
        '15 mins' : '15minute',
        '30 mins' : '30minute',
        '1 hour' : '60minute',
        '2 hour': '2hour',
        '3 hour': '3hour'
    }
    
  index= individual_values[f'index_{strategy_no}']
  def logFileInit():
        dir=fr".\\{folder}\\"
      
        if not os.path.exists(dir):
                    print('creating logfile directory')
                    os.makedirs(dir)
                    
        nowtime=time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
        logFilename=dir+'Index_BUY_Algo_'+nowtime+'_Strategy_'+str(strategy_no)+'_Bucket_'+str(premium_bucket+1)+'_'+index+'.log'
        global logFile
        
        logFile = open(logFilename, "a+")
        sys.stdout = logFile
        print('Here after all the logs will be written to a file:',logFilename,'\n')
        #print(logFilename)
        return logFile
  global logFile,Order_Identifier
  logFile = None
  logFile=logFileInit()
  nowtimee = datetime.datetime.now()
 
  entry_time = individual_values[f'entry_time_{strategy_no}']
  print('Entry Time:',entry_time)
  logFile.flush(); os.fsync(logFile.fileno())

  current_time = nowtimee.time()
  if current_time<entry_time:
       print(f"Algo is going to wait till the ALGO Entry time {entry_time} is reached.")
       logFile.flush(); os.fsync(logFile.fileno())

  while current_time <= entry_time:
      #time.sleep(2)  
      nowtimee = datetime.datetime.now()
      current_time = nowtimee.time()
  
  if individual_values[f'enable_strategy_{strategy_no}']=="Yes":
   ##we have to take premium from strike price algos and call monitor options functions
   #print('Thread started for Strategy',strategy_no)
   strike_prices_algo = [individual_values[f'premium_range1_{strategy_no}'],individual_values[f'premium_range2_{strategy_no}'],individual_values[f'premium_range3_{strategy_no}'],individual_values[f'premium_range4_{strategy_no}']]
   strike_prices_algo_bkp = strike_prices_algo.copy()
   #print('strike_prices_algo 123:',strike_prices_algo,'premium_bucket:',premium_bucket)
   if strike_prices_algo_bkp[premium_bucket] != "Not Applicable":
    strike_prices_algo = []
    strike_prices_algo.append(strike_prices_algo_bkp[premium_bucket])
    strike_prices_algo = [price for price in strike_prices_algo if not (isinstance(price, float) and math.isnan(price)) and not isinstance(price, str)]
    #print('strike_prices_algo:',strike_prices_algo)
    print('This bucket is enabled Strategy & Premium bucket:',strategy_no,premium_bucket+1,'and premium:',strike_prices_algo,'Current Time:',datetime.datetime.now())
    Order_Identifier = 'Algo_' + str(strategy_no) + '_' + str(premium_bucket+1)
    file_lock = threading.Lock()      
    events_algo = {strike_price: threading.Event() for strike_price in strike_prices_algo}
    #print('events_algo:',events_algo)
    for event in events_algo.values():
        event.set()
  
    
    def calculate_last_bearish_candle_with_rsi(df, logFile,lower_limit_value):
            df = df.iloc[::-1]
            for row in df.itertuples():
                if (row.open > row.close) and (row.rsi <lower_limit_value):
                    result = row.low
                    return result
    def calculate_last_bearish_candle(df,logFile):
        df = df.iloc[::-1]
    
        # find the first row with 'open' value greater than 'close'
        for row in df.itertuples():
            if row.open > row.close:
                result = row.low
                return result
                                
    

    def heikin_ashi_df(rel_df,logFile,last_row=False):
        #print('Heiken before:',rel_df.tail(2))
        if not last_row:
            rel_df = rel_df.drop(rel_df.index[-1])
        

        #print('Heiken before:',rel_df.tail(2))
        date_list = []
        date_list = rel_df.date.tolist();#print(date_list);exit()
        pd.options.mode.chained_assignment = None
        HAdf = rel_df[['open', 'high', 'low', 'close']]
        HAdf['close'] = round(((rel_df['open'] + rel_df['high'] + rel_df['low'] + rel_df['close'])/4),2)
    
        for i in range(len(rel_df)):
            if i == 0:
                HAdf.iat[0,0] = round(((rel_df['open'].iloc[0] + rel_df['close'].iloc[0])/2),2)
            else:
                HAdf.iat[i,0] = round(((HAdf.iat[i-1,0] + HAdf.iat[i-1,3])/2),2)
        
        HAdf['high'] = HAdf.loc[:,['open', 'close']].join(rel_df['high']).max(axis=1)
        HAdf['low'] = HAdf.loc[:,['open', 'close']].join(rel_df['low']).min(axis=1)
        HAdf['date'] = date_list
        cols = HAdf.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        HAdf = HAdf[cols];       
        return HAdf
    
    def heikin_ashi(rel_df,logFile,new=False) :
        if not new:   
            df= heikin_ashi_df(rel_df,logFile)
        if new:
            df= heikin_ashi_df(rel_df,logFile,True)
        
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
        logFile.flush(); os.fsync(logFile.fileno())
    
        return C1,C2,O2,H2,L2,RO,df,P1_time
    
    def calculate_last_rsi(ndf, time_period,logFile):
        df= heikin_ashi_df(ndf,logFile)
        
        logFile.flush(); os.fsync(logFile.fileno());
    
        df['rsi']  = talib.RSI(df['close'], timeperiod=time_period)
        #print('rsi calculation time',datetime.datetime.now())
        logFile.flush(); os.fsync(logFile.fileno());
        last_row_rsi = df['rsi'].iloc[-1]
        previous_row_rsi= df['rsi'].iloc[-2]
    
        return last_row_rsi,previous_row_rsi,df
    
    def get_historic_data(name, timeframe, delta, Instrument_type,logFile,From):
        while True:
            try:
                if 'FUT' in name:
                 timeframe = bnf_tf1
                elif name in ['NIFTY MID SELECT','BANKEX','SENSEX']:
                 timeframe = bnf_tf1                 
                else:
                 timeframe = opt_tf1
                to_date = datetime.datetime.now()
                print("Extracting historic data In:",name, timeframe, delta, Instrument_type,'From:',From,to_date)
                logFile.flush(); os.fsync(logFile.fileno())
                delta = 6
                from_date = to_date - datetime.timedelta(days=int(delta))
                if name in token_dict:
                 #print("Exists")
                 token = token_dict[name]
                else:
                 #print("Does not exist")
                 if name=="SENSEX" or name=="BANKEX":
                    token = kite.ltp(['BSE'+':' + name])['BSE'+':' + name]['instrument_token']
                    token_dict[name] = token
                 elif name=='NIFTY MID SELECT':
                    token = kite.ltp(['NSE'+':' + name])['NSE'+':' + name]['instrument_token']
                    token_dict[name] = token

                 elif name.startswith("SENSEX") or name.startswith("BANKEX"):
                    token = kite.ltp(['BFO'+':' + name])['BFO'+':' + name]['instrument_token']
                    token_dict[name] = token
                 else:
                    token = kite.ltp([Instrument_type+':' + name])[Instrument_type+':' + name]['instrument_token']
                    token_dict[name] = token
                     
                #print(token_dict)
                new_data = kite.historical_data(token, from_date, to_date, timeframe)
                new_data_df = pd.DataFrame(new_data)         
                new_data_df['date'] = new_data_df['date'].dt.tz_localize(None)
                #print('historic data time',datetime.datetime.now())
                logFile.flush(); os.fsync(logFile.fileno());
                return new_data_df
            except Exception as e:
                print(f"Exception occurred on get_historic_data {name}, retrying...: " + str(e))
                time.sleep(1) 
        

    def get_data(name, timeframe, delta, Instrument_type,logFile,rsi_length):
                
        new_data_df= get_historic_data(name, timeframe, delta, Instrument_type,logFile,1)
        # if len(new_data_df.index) >0:
        C1,C2,O2,H2,L2,RO,hdf,p1_time = heikin_ashi(new_data_df,logFile)
        RSI,prev_rsi,rsi_df = calculate_last_rsi(new_data_df,rsi_length,logFile)
        #print('timeframe',timeframe)
        #print("get data time",datetime.datetime.now()) 
        #print(C1,C2,O2,H2,L2,RO,RSI,p1_time)
    
        logFile.flush(); os.fsync(logFile.fileno());          
        return C1,C2,O2,H2,L2,RSI,hdf,prev_rsi,p1_time,rsi_df
    
    def last_2_candles_low_sl(sl_df,logFile):
       
        sl_df= sl_df.tail(2)
        first_row_low = sl_df.iloc[0]['low']
        second_row_low = sl_df.iloc[1]['low']
    
        print("Low value of the first row:", first_row_low)
        print("Low value of the second row:", second_row_low)
        logFile.flush(); os.fsync(logFile.fileno());          
    
        lowest_low = sl_df['low'].min()
        print('last 2 candles  low:',datetime.datetime.now() )
        logFile.flush(); os.fsync(logFile.fileno());          
    
        return lowest_low
    
    def get_strikeprice(index,premium):
        
            ##get Future
            Index_to_TAKE_TRADE = index #'BANKNIFTY'/'NIFTY'
            Expiry_to_use = 'WEEKLY';ins_file_nm = ''
            if Index_to_TAKE_TRADE == 'NIFTY':
             df12 = pd.read_csv('instruments_NY.csv')
             ins_file_nm = 'instruments_NY.csv'
            elif Index_to_TAKE_TRADE == 'BANKNIFTY':
             df12 = pd.read_csv('instruments_BN.csv')
             ins_file_nm = 'instruments_BN.csv'
            elif Index_to_TAKE_TRADE == 'FINNIFTY':
             df12 = pd.read_csv('instruments_FN.csv')
             ins_file_nm = 'instruments_FN.csv'
            elif Index_to_TAKE_TRADE == 'SENSEX':
             df12 = pd.read_csv('instruments_SEN.csv')
             ins_file_nm = 'instruments_SEN.csv'
            elif Index_to_TAKE_TRADE == 'MIDCPNIFTY':
             df12 = pd.read_csv('instruments_MID.csv')
             ins_file_nm = 'instruments_MID.csv'
            elif Index_to_TAKE_TRADE == 'BANKEX':
             df12 = pd.read_csv('instruments_BAN.csv')
             ins_file_nm = 'instruments_BAN.csv'
            
            Expiry_dt1,MonthlyExpiry1 = Strike_Selection.Expiry_selection(Expiry_to_use,'','NO','NO',Index_to_TAKE_TRADE,ins_file_nm)

            df12['expry'] = df12["expiry"].astype(str) #'2021-08-05' 'BANKNIFTY'
            
            df12 = df12[((df12.name == Index_to_TAKE_TRADE) & ((((df12.segment == 'NFO-FUT') & (df12.exchange == 'NFO')) | ((df12.segment == 'BFO-FUT') & (df12.exchange == 'BFO'))) & (df12.expry == MonthlyExpiry1)))]
            val = df12.iloc[-1];future_nm = val['tradingsymbol'];print('Future Contract is:',future_nm)
            ATM_selection_Method = 'INDEX_CHART' #INDEX_CHART/FUTURE_CHART
            #ATM_selection_Method = 'Index';
            Index_to_TAKE_TRADE = index #'BANKNIFTY'/'NIFTY'
            ATM_Strike = 'ATM'
            OTM_PLUS_MINUS = '+1' ##if +1 then OTM
    
             # WEEKLY,MONTHLY
            Manual_expiry = str('2021-01-20')
            Expiry_Day_use_Expiry_to_USE_W = 'NO' 
            Expiry_Day_use_Expiry_to_USE_M = 'NO'
            FIFTY_strike_To_Consider = 'YES'
            Strike_Selection_Method = 'Less_than_Premium_Based' #Premium_Based , Strike_Based , Less_than_Premium_Based
    
            OTM_BUY_Lower_Limit = premium
            OTM_BUY_Upper_Limit = OTM_BUY_Lower_Limit + 200
            if True:
                        Spot_To_Use = 0
                        OTM_BUY_Upper_Limit2 = OTM_BUY_Lower_Limit + 200
                        #Inhouse 2; To check for the adjustment opportunity.
                        stocks_list9,buy_stocks_list9,sell_stocks_list_INITIAL_temp,strike_price9,future_contract_list9,Entry_spot_Check_Time = Strike_Selection.strike_selection(Expiry_to_use,Manual_expiry,Expiry_Day_use_Expiry_to_USE_W,Expiry_Day_use_Expiry_to_USE_M,OTM_PLUS_MINUS,ATM_Strike,Index_to_TAKE_TRADE,ATM_selection_Method,FIFTY_strike_To_Consider,OTM_BUY_Lower_Limit,OTM_BUY_Upper_Limit2,Strike_Selection_Method,Spot_To_Use,kite,ins_file_nm);
                        print(sell_stocks_list_INITIAL_temp)
            return future_nm,sell_stocks_list_INITIAL_temp
    
    def last_bearish_candle_low(name, timeframe, delta, Instrument_type, logFile):
        df= get_historic_data(name, timeframe, delta, Instrument_type,logFile,2)
        C1,C2,O2,H2,L2,RO,hdf,p1_time = heikin_ashi(df,logFile)
          
        df = hdf.iloc[::-1]
        for row in df.itertuples():
            if row.open > row.close:
                result = row.low
                return result
        
    
    def get_close_and_rsi(name, timeframe, delta, Instrument_type, logFile,rsi_length):
        new_data_df= get_historic_data(name, timeframe, delta, Instrument_type, logFile,3)
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df, logFile)
        RSI, previous_rsi,rsi_df= calculate_last_rsi(new_data_df,rsi_length,logFile)
        print('get close and rsi:',datetime.datetime.now() )
        logFile.flush(); os.fsync(logFile.fileno());           
        return C2,O2,RSI,previous_rsi,L2,p1_time
    
    def bullish_condition(bull_condt,i,order_executed_price,trade,local_quantity,all_exited,sl_order_id,isl_value,name, timeframe, delta, Instrument_type, logFile,rsi_length):
        new_data_df= get_historic_data(name, timeframe, delta, Instrument_type, logFile,6)
        df= heikin_ashi_df(new_data_df,logFile)
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

            row1 = bullish_rows.iloc[0][individual_values[f'bullish_condition_{i}']]
            print("p3 candle selected value is",bullish_rows.iloc[0][individual_values[f'bullish_condition_{i}']])
            row2 = bullish_rows.iloc[1][individual_values[f'bullish_condition_{i}']]
            print("p2 candle selected value is",bullish_rows.iloc[1][individual_values[f'bullish_condition_{i}']])
            
            row3 = bullish_rows.iloc[2][individual_values[f'bullish_condition_{i}']]
            print("p1 candle selected value is",bullish_rows.iloc[2][individual_values[f'bullish_condition_{i}']])


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
                    sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                    print("Shifting isl/tsl value")
                    isl_value=lowest_low

                              
                logFile.flush(); os.fsync(logFile.fileno())
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
        
    def checking_complete_status(opt_low,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name,opt_tf,logFile,trade):
    
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
                ltp= get_ltp(name,logFile)
    
                if sl_price>ltp:
                    print("sl price is greater than ltp market moved in opp direction before placing sl so quitting the taken trade")
                    logFile.flush(); os.fsync(logFile.fileno())
                    
                    order_type= 'market'
                    carry= individual_values[f'product_type_{i}']
                    limit_value=0                                 
                    oid=orderplacement.order_placement(name,'SELL',qty,logFile,kite,Order_Identifier,order_type,carry,limit_value)
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                    if status=="COMPLETE":
                        all_exited=True
                        
                        return all_exited,sl_order_id,local_quantity,status,isl_value
    
                else:
                    carry= individual_values[f'product_type_{i}']
                    if carry=='normal':
                        product_type=kite.PRODUCT_NRML
                    elif carry=='intraday':
                        product_type=kite.PRODUCT_MIS
                    sl_order_id,all_exited = orderplacement.Zerodha_place_sl_order_with_verification(carry,all_exited,name,qty,kite.ORDER_TYPE_SL,product_type,price,kite.TRANSACTION_TYPE_SELL,kite.EXCHANGE_NFO,kite.VARIETY_REGULAR,sl_price,kite.VALIDITY_DAY,logFile,kite,Order_Identifier)        
                    

                print("sl order id", sl_order_id)            
                print('bearish candle sl is', sl_price)
                logFile.flush(); os.fsync(logFile.fileno())
                return all_exited,sl_order_id,local_quantity,status,isl_value
            
    
        else:
            return all_exited,sl_order_id,local_quantity,status,isl_value
            
    def check_candle_condition(opt_low,trade,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name,opt_tf,logFile):
    
        if status=='COMPLETE':
            all_exited,sl_order_id,local_quantity,status,isl_value= checking_complete_status(opt_low,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name,opt_tf,logFile,trade)        
            return all_exited,sl_order_id,local_quantity,isl_value
        
        elif status!='COMPLETE':
            if (status== 'REJECTED' or status=='CANCELLED'):
                print("condition met but order placement failed due to some reasons placing orders again")
                logFile.flush(); os.fsync(logFile.fileno())
    
                for i in range(1,100):
                    oid= orderplacement.modify_mkt_order(sl_order_id,qty)
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                    all_exited,sl_order_id,local_quantity,status,isl_value= checking_complete_status(opt_low,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name,opt_tf,logFile,trade)        
                    if status=='COMPLETE':
                        return all_exited,sl_order_id,local_quantity,isl_value
                    else:              
                        time.sleep(1)
    
                local_quantity= local_quantity+qty                
                return all_exited,sl_order_id,local_quantity,isl_value
                        
            elif status=='PENDING': 
                print("condition met but order placement pending due to some reasons checking again")
                logFile.flush(); os.fsync(logFile.fileno())
                for i in range(1,100):
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
                    all_exited,sl_order_id,local_quantity,status,isl_value= checking_complete_status(opt_low,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name,opt_tf,logFile,trade)        
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
                logFile.flush(); os.fsync(logFile.fileno()); 
                local_quantity=local_quantity-quant_to_sell          
    
                return quant_to_sell,local_quantity
            else:
                lots_to_sell = math.ceil(lots / 2)
                quant_to_sell= int(lots_to_sell*lot_size)
                print('quant_to_sell', quant_to_sell)
                logFile.flush(); os.fsync(logFile.fileno()); 
                local_quantity=local_quantity-quant_to_sell                    
    
                return quant_to_sell,local_quantity
    
    # def set_sl(order_exec_price,name,opt_tf,logFile,i,local_quantity,sl_order_id,all_exited):
    
    #     qty,local_quantity = qty_to_exit(name,logFile,i,local_quantity)
        
    #     print(' exiting 50 percent at market:',qty,'available qty:',local_quantity)
    #     logFile.flush(); os.fsync(logFile.fileno()); 
    
    #     if individual_values[f'index_{i}']=='BANKNIFTY':
    #         lot_size=individual_values[f'bnf_lot_size_{i}']
    #     if individual_values[f'index_{i}']=='NIFTY':
    #         lot_size=individual_values[f'nifty_lot_size_{i}']
    #     if individual_values[f'index_{i}']=='FINNIFTY':
    #         lot_size=individual_values[f'finnifty_lot_size_{i}']
    
    #     bearish_sl=True
    #     oid= orderplacement.modify_mkt_order(sl_order_id,qty)
    #     order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
    #     all_exited,sl_order_id,local_quantity,isl_value= check_candle_condition(trade,isl_value,bearish_sl,order_exec_price,all_exited,local_quantity,qty,lot_size,status,sl_order_id,name,opt_tf,logFile)
    #     return all_exited,sl_order_id,local_quantity
       
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



    def get_supertrend(df,logFile,i):
        # df=get_historic_data(k_ts,bnf_tf,delta,itype,logFile,5)
        # hdf=heikin_ashi_df(df,logFile)
        supertrend_period = individual_values[f'length_{i}']
        supertrend_multiplier = individual_values[f'factor_{i}']
        sdf = SuperTrend(df,int(supertrend_period),int(supertrend_multiplier))
        df= sdf.tail(1)
        supertrend_value= df['STX'].values[0]
        print('super trend is',supertrend_value)
        return supertrend_value

    def data_analysis(i,index_future,pe_tradingsymbol,ce_tradingsymbol,logFile,bnf_tf,opt_tf):
                
        ce_ts=ce_tradingsymbol
        pe_ts=pe_tradingsymbol
        k_ts= index_future
        rsi_length= individual_values[f'rsi_{i}']
        #print("rsi length is",rsi_length)  
        #print('symbols', ce_ts,pe_ts,k_ts)
        logFile.flush(); os.fsync(logFile.fileno())
    
        CC1,CC2,CO2,CH2,CL2,CRSI,cdf,prev_crsi,callp1_time,ce_rsi_df= get_data(ce_ts,opt_tf,4,'NFO',logFile,rsi_length)               
        PC1,PC2,PO2,PH2,PL2,PRSI,pdf,prev_prsi,putp1_time,pe_rsi_df= get_data(pe_ts,opt_tf,4,'NFO',logFile,rsi_length)
        C1,C2,O2,H2,L2,RSI,fdf,prev_rsi,candle_p1_time,fut_rsi_df= get_data(k_ts,bnf_tf,4,'NFO',logFile,rsi_length)
        supertrend_value=get_supertrend(fdf,logFile,i)
        call_supertrend_value=get_supertrend(cdf,logFile,i)
        put_supertrend_value=get_supertrend(pdf,logFile,i)

        print("Candle data Extration completion time",datetime.datetime.now())
        logFile.flush(); os.fsync(logFile.fileno())
        return C1,C2,O2,H2,L2,RSI,prev_rsi,candle_p1_time,CC1,CC2,CO2,CH2,CL2,CRSI,prev_crsi,callp1_time,PC1,PC2,PO2,PH2,PL2,PRSI,prev_prsi,putp1_time,cdf,pdf,supertrend_value,call_supertrend_value,put_supertrend_value,ce_rsi_df,pe_rsi_df,fut_rsi_df
        #return C1,C2,O2,H2,L2,RSI,prev_rsi,CC1,CC2,CO2,CH2,CL2,CRSI,prev_crsi,PC1,PC2,PO2,PH2,PL2,PRSI,prev_prsi,cdf,pdf,candle_p1_time
    
    def option_ltp(tradingsymbol,logFile):
        for i in range(0,100):
            try:
                if tradingsymbol.startswith("SENSEX") or tradingsymbol.startswith("BANKEX"):
                    ltp = kite.ltp(['BFO:'+ tradingsymbol])['BFO:'+ tradingsymbol]['last_price']
                else:
                    ltp = kite.ltp(['NFO:'+ tradingsymbol])['NFO:'+ tradingsymbol]['last_price']
                print('retuning ltp of '+ tradingsymbol)
                print(time.localtime())
                logFile.flush(); os.fsync(logFile.fileno());
                
                return(ltp)
            except Exception as e:
                print('Exception on option_ltp:'+ str(e))
                logFile.flush(); os.fsync(logFile.fileno());
                time.sleep(1)
                if i==99:
                    print(tradingsymbol+'ltp couldnt be fetched after 10 tries, so exiting with ZERO')
                    ltp=0
                    return ltp
                    
        print(tradingsymbol+'ltp couldnt be fetched,trying again due to __'+ str(e))
    
    def get_ltp(tradingsymbol,logFile):
        ltp=option_ltp(tradingsymbol,logFile)
        print('ltp is:',ltp)
        logFile.flush(); os.fsync(logFile.fileno())
        return ltp
         
    
    def get_ltp_and_current_opt_open(tradingsymbol,logFile,opt_tf):
        ltp=option_ltp(tradingsymbol,logFile)
        print('Inside get_ltp_and_current_opt_open: ltp is:',ltp,opt_tf)
        logFile.flush(); os.fsync(logFile.fileno())
    
        new_data_df= get_historic_data(tradingsymbol, opt_tf, 2, 'NFO',logFile,4)
    
        logFile.flush(); os.fsync(logFile.fileno());
    
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df,logFile)
        print('returning ltp for trade time',datetime.datetime.now())
        logFile.flush(); os.fsync(logFile.fileno());
        
        return ltp,RO

    def get_ltp_and_current_opt_high(tradingsymbol,logFile,opt_tf):
        ltp=option_ltp(tradingsymbol,logFile)
        print('Inside get_ltp_and_current_opt_high: ltp is:',ltp,opt_tf)
        logFile.flush(); os.fsync(logFile.fileno())   
        new_data_df= get_historic_data(tradingsymbol, opt_tf, 2, 'NFO',logFile,4)    
        logFile.flush(); os.fsync(logFile.fileno());    
        C1,C2,O2,H2,L2,RO,df,p1_time = heikin_ashi(new_data_df,logFile,True)        
        print('returning ltp for trade time',datetime.datetime.now())
        logFile.flush(); os.fsync(logFile.fileno());
        
        return ltp,H2
    
    def read_excel(logFile):
        
      while True:  
        try:
            df = pd.read_excel('Index_BUY_Algo_Configurations.xlsm')
    
            df[df.columns[0]] = df[df.columns[0]].str.lower().replace(' ', '_', regex=True)
            data = {}
    
            for i, row in df.iterrows():
                if i < 1: 
                    continue
                key = row[df.columns[0]]
                data[key] = list(row[1:])
                
            individual_values = {}
    
            for key in data:
                for i, val in enumerate(data[key], start=1):
                    individual_values[f'{key}_{i}'] = val
            
            return individual_values
        
        except Exception as e:
            print(f"An error occurred while reading the file: {e}. Retrying in a few seconds.")
            logFile.flush(); os.fsync(logFile.fileno())
            time.sleep(2)
    
    def remove_one_instance_from_file(filename, logFile, content, max_retries=5):
        retry_count = 0
    
        while retry_count < max_retries:
            with file_lock:  # Acquiring the lock to ensure thread-safe file operations
                try:
                    with open(filename, "r") as file:
                        lines = file.readlines()
                    
                    if content + "\n" in lines:
                        lines.remove(content + "\n")
                        print(f'{content} got removed')
                        logFile.flush(); logFile.fileno()
                        
                        with open(filename, "w") as file:
                            file.writelines(lines)
                        
                    else:
                        print(f"Content '{content}' not found in the file.")
                        
                    break  # Exit the loop in both cases (content found or not)
                    
                except Exception as e:
                    print(f"An error occurred while modifying the file: {e}. Retrying...")
                    retry_count += 1  # Increment the retry count
                    time.sleep(2)  # Wait for 2 seconds before retrying
                
        if retry_count == max_retries:
            print("Max retry attempts reached. Could not modify the file.")


    def target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity):
        
        if individual_values[f'target_split_{i}']=="enable":
            print("Target split is enabled")
            first_target_trailing=False
            second_target_trailing=False
            second_sl=False
            if first_target and not first_target_trailing:
                first_targett= individual_values[f'first_target_{i}']
                target= order_exec_price+ (first_targett*order_exec_price)
                print("First target is",target)
                logFile.flush(); os.fsync(logFile.fileno())
                

                if o_ltp> target or high> target:
                    print("o_ltp is",o_ltp)
                    print("high is",high)
                    print("first target is",target)

                    print("First target has been achieved")
                    first_qty,local_quantity=qty_to_exit(name,logFile,i,local_quantity)
                    first_target_percent= individual_values[f'first_target_trailing_{i}']
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
                    
                    logFile.flush(); os.fsync(logFile.fileno())
                    first_target=False
                    
            
                    if all_exited:
                        second_target=True
                        all_exited=False
                        first_target_trailing=False
                        first_target=False
                        second_sl=True
            
            if first_target_trailing:
                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',sl_order_id)
                print("first targetr order status is not completed entering into while loo[p]")
                first_target_percent= individual_values[f'first_target_trailing_{i}']
                while status !="COMPLETE":
                    o_ltp=option_ltp(name,logFile)
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
                    
                    order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',sl_order_id)
                    
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
            
                second_targett= individual_values[f'second_target_{i}']
                target= order_exec_price+ (second_targett*order_exec_price)
                

                print("second target is",target)
                logFile.flush(); os.fsync(logFile.fileno())
                if o_ltp> target or high > target :
                    print("second target has been achieved ")
                    print("o_ltp is",o_ltp)
                    print("high is",high)
                    print("second target is",target)
                    second_target_percent= individual_values[f'second_target_trailing_{i}']
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
                    second_target_trailing_point= target + (target*individual_values[f'second_target_trailing_{i}'])
                    print("second target trailing point is",second_target_trailing_point)
                    
                    print("sl has been modified for half quantity")
                    
                    logFile.flush(); os.fsync(logFile.fileno())
                    
                    second_target=False
            
                    if all_exited:
                        second_target_trailing=False
                        all_exited=True
                        
            
            if second_target_trailing:
                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,orderexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',sl_order_id)
                print("second targetr order status is not completed entering into while loo[p]")
                second_target_percent= individual_values[f'second_target_trailing_{i}']
                while status !="COMPLETE":
                    o_ltp=option_ltp(name,logFile)
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

        
        elif individual_values[f'target_split_{i}']=="disable":
            print("Target split is disabled") 
            target_percent= individual_values[f'target_{i}']
            target= order_exec_price+ (target_percent*order_exec_price)

            print('target is',target)
            logFile.flush(); os.fsync(logFile.fileno())

            if o_ltp> target or high > target:  ##logic to exit the full quantities
                oid= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
                time.sleep(5)
                #status= orderplacement.get_orderstatus(oid,logFile)
                order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,order_xec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',oid)
            
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
        logFile.flush(); os.fsync(logFile.fileno())
    
        no_of_times= individual_values[f'single_candle_condition_{i}']  
        o_ltp,current_candle_open=get_ltp_and_current_opt_open(name,logFile,opt_tf)
        single_candle_target= (no_of_times*current_candle_open) + current_candle_open
        print('no of times:',no_of_times, 'o_ltp;',o_ltp, 'single candle target', single_candle_target, 'current candle open',current_candle_open)
        
        logFile.flush(); os.fsync(logFile.fileno())
        ###target checking###
        if o_ltp> single_candle_target:
            print("single candle target achieved")
            logFile.flush(); os.fsync(logFile.fileno())
            selected_percent= individual_values[f'tsl_of_scc_{i}']   
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
                logFile.flush(); os.fsync(logFile.fileno())
                trailing_sl_movement_percent= individual_values[f'after_scc_x_pct_price_move_{i}']  ##get the values from excel
                trailing_sl_movement= (trailing_sl_movement_percent * current_candle_open)
                trailing_sl_target= single_candle_target+ trailing_sl_movement
                tsl_percent= individual_values[f'after_scc_y_pct_trailing_move_{i}']
                single_candle_tsl= (current_candle_open * tsl_percent)
                tslmovement= True
                print('trailing sl target', trailing_sl_target,'single candle tsl',single_candle_tsl)
                logFile.flush(); os.fsync(logFile.fileno())
    
        if tslmovement: 
            print("entered into if tslmovement")
            logFile.flush(); os.fsync(logFile.fileno())
    
            if o_ltp> trailing_sl_target: 
                print("trailing sl target achieved") 
                logFile.flush(); os.fsync(logFile.fileno())
    
                tsl= single_candle_sl+single_candle_tsl
                print('tsl is', tsl)
                logFile.flush(); os.fsync(logFile.fileno())
    
                if (((o_ltp-tsl)> (0.03*o_ltp)) and (tsl > isl_value)):
                    print("if ((o_ltp-tsl)> (0.1*o_ltp)):") 
                    print(" shifting x y based tsl")
                    logFile.flush(); os.fsync(logFile.fileno())
                    qty=local_quantity
                    sl_order_id,all_exited= orderplacement.def_modify_sl_order_for_sell(sl_order_id,tsl,qty,all_exited) 
                    isl_value=tsl
                    
                    logFile.flush(); os.fsync(logFile.fileno())
    
                else:
                    print(f'single candle sl shifting already happened and {tsl} value is much closer to ltp so tsl not happened')     
                    logFile.flush(); os.fsync(logFile.fileno())
                    
         
                trailing_sl_target= trailing_sl_target+trailing_sl_movement
        
        print(" trailing sl revised target",trailing_sl_target)
        logFile.flush(); os.fsync(logFile.fileno())              
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

       
        if (individual_values[f'tsl1_{i}'][3:]== 'Exit half and remaining at bearish candle low' 
            or individual_values[f'tsl2_{i}'][3:]== 'Exit half and remaining at bearish candle low' 
            or individual_values[f'tsl1_{i}'][3:]== 'Exit at market price and half trailing with bearish candle low sl'
            or individual_values[f'tsl2_{i}'][3:]== 'Exit at market price and half trailing with bearish candle low sl'
            or individual_values[f'tsl1_{i}'][3:]== 'Full trailing and half trailing with bearish candle low sl'
            or individual_values[f'tsl2_{i}'][3:]== 'Full trailing and half trailing with bearish candle low sl'):
            bearish_sl=True
            print("entered into Exit half and remaining at bearish candle low")
            logFile.flush(); os.fsync(logFile.fileno())
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
        logFile.flush(); os.fsync(logFile.fileno())
    
        if last_candle_close < order_exec_price: ##exit all quantity logic
            print("p1_candle_low < order_exec_price ",'last low is',last_candle_close,'order_executed price is',order_exec_price)
            print(" condition satisfied closing the trade")
            logFile.flush(); os.fsync(logFile.fileno())
    
            exit_all= orderplacement.modify_mkt_order(sl_order_id,local_quantity)
            #status=orderplacement.get_orderstatus(exit_all,logFile)
            order_id3,tradingsymbol12,get_current_TGTorSL_type,price1,trigger_price1,status,filled_Quantity1,Quantity1 ,transaction_type1,ordeexec_price,exchange_timestamp1 = orderplacement.get_open_order_details(logFile,'',exit_all)
            if status=='COMPLETE':
                all_exited= True
                print("exited all")
                
                logFile.flush(); os.fsync(logFile.fileno())
    
                local_quantity=0
                return all_exited,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,local_quantity,ssl
                                           
                
    
        if True and not all_exited:
            sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,ssl,all_exited,isl_value=single_candle_condition_checking(isl_value,trade,all_exited,ssl,local_quantity,i,name,opt_tf,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl)
            o_ltp=get_ltp(name,logFile)
            ltpp,high=get_ltp_and_current_opt_high(name,logFile,opt_tf)

            all_exited,local_quantity,first_target,second_target,sl_order_id=target_checking(high,first_target,second_target,Order_Identifier,isl_value,name,trade,i,o_ltp,logFile,order_exec_price,all_exited,sl_order_id,local_quantity)
        
        
        return all_exited,sl_order_id,tslmovement,trailing_sl_movement,trailing_sl_target,single_candle_tsl,single_candle_sl,local_quantity,ssl
    
    def get_quantity(ltp,i,logFile):
        print('Calculating Quantity Based on the available/allowed Capital!!')
        if individual_values[f'cbt_{i}']!='no':
            logFile.flush(); os.fsync(logFile.fileno())
            given_capital= individual_values[f'capital_to_deploy_{i}']
            percent_to_take_trade= 1
            cpt = float(percent_to_take_trade * given_capital)
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
            print('percent_to_take_trade:',percent_to_take_trade,'given_capital:',given_capital,individual_values[f'index_{i}'],"lot size is", lot_size)    
            logFile.flush(); os.fsync(logFile.fileno())
            cost_per_lot= ltp*lot_size
    
            if cpt< cost_per_lot:
                print("\n\nAlert!!: Not enough Capital to take a single lot!!. so trade wont be taken!!")
                return 0
            else:
                lots = cpt // cost_per_lot  
                qty = int(lots * lot_size)
                print("calculated quantity before checking the max llot size is", qty)
                df = pd.read_excel('Index_BUY_Algo_Configurations.xlsm', sheet_name='Credentials', header=None)
                data_dict = dict(zip(df[0], df[1]))
                nifty_max_qty=data_dict['NiftyMaxQty']
                bn_max_qty= data_dict['BankniftyMaxQty']
                fn_max_qty=data_dict['FinniftyMaxQty']
                sensex_max_qty=data_dict['SensexMaxQty']
                bankex_max_qty=data_dict['BankexMaxQty']
                midcpnifty_max_qty=data_dict['MidcpniftyMaxQty']
                # print(f"Max allowed qty of nifty is {nifty_max_qty}, banknifty is {bn_max_qty}, finnifty is {fn_max_qty}")
                print(f"Max allowed qty of nifty is {nifty_max_qty}, banknifty is {bn_max_qty}, finnifty is {fn_max_qty}, sensex is {sensex_max_qty},bankex is{bankex_max_qty},midcap nifty is {midcpnifty_max_qty}")

                
                if  individual_values[f'index_{i}']=='BANKNIFTY':
                    if qty > bn_max_qty:
                        print(f'{qty} is greater than allowed lot size qty of banknifty {bn_max_qty}')
                        qty=bn_max_qty
                
                elif individual_values[f'index_{i}']=='NIFTY':
                    if qty > nifty_max_qty:
                        print(f'{qty} is greater than allowed lot size qty of nifty {nifty_max_qty}')

                        qty=nifty_max_qty
                
                elif individual_values[f'index_{i}']=='FINNIFTY':
                    if qty > fn_max_qty:
                        print(f'{qty} is greater than allowed lot size qty of finifty {fn_max_qty}')
                        qty=fn_max_qty

                elif individual_values[f'index_{i}']=='SENSEX':
                    if qty > sensex_max_qty:
                        print(f'{qty} is greater than allowed lot size qty of finifty {sensex_max_qty}')
                        qty=sensex_max_qty
            
                elif individual_values[f'index_{i}']=='BANKEX':
                    if qty > bankex_max_qty:
                        print(f'{qty} is greater than allowed lot size qty of finifty {bankex_max_qty}')
                        qty= bankex_max_qty

                elif individual_values[f'index_{i}']=='MIDCPNIFTY':
                    if qty > midcpnifty_max_qty:
                        print(f'{qty} is greater than allowed lot size qty of finifty {midcpnifty_max_qty}')
                        qty=midcpnifty_max_qty
                           
                print('Identified Qty:',qty);logFile.flush(); os.fsync(logFile.fileno())
                logFile.flush(); os.fsync(logFile.fileno())
                return qty
        else:
            qty = int(individual_values[f'quantity_{i}'])
            print('Identified Qty:',qty);logFile.flush(); os.fsync(logFile.fileno())
            print("given quantity before checking the max llot size is", qty)
            df = pd.read_excel('Index_BUY_Algo_Configurations.xlsm', sheet_name='Credentials', header=None)
            data_dict = dict(zip(df[0], df[1]))
            nifty_max_qty=data_dict['NiftyMaxQty']
            bn_max_qty= data_dict['BankniftyMaxQty']
            fn_max_qty=data_dict['FinniftyMaxQty']
            sensex_max_qty=data_dict['SensexMaxQty']
            bankex_max_qty=data_dict['BankexMaxQty']
            midcpnifty_max_qty=data_dict['MidcpniftyMaxQty']
            print(f"Max allowed qty of nifty is {nifty_max_qty}, banknifty is {bn_max_qty}, finnifty is {fn_max_qty}, sensex is {sensex_max_qty},bankex is{bankex_max_qty},midcap nifty is {midcpnifty_max_qty}")
            

            if  individual_values[f'index_{i}']=='BANKNIFTY':
                if qty > bn_max_qty:
                    print(f'{qty} is greater than allowed lot size qty of banknifty {bn_max_qty}')
                    qty=bn_max_qty
            
            elif individual_values[f'index_{i}']=='NIFTY':
                if qty > nifty_max_qty:
                    print(f'{qty} is greater than allowed lot size qty of nifty {nifty_max_qty}')

                    qty=nifty_max_qty
            
            elif individual_values[f'index_{i}']=='FINNIFTY':
                if qty > fn_max_qty:
                    print(f'{qty} is greater than allowed lot size qty of finifty {fn_max_qty}')
                    qty=fn_max_qty

            elif individual_values[f'index_{i}']=='SENSEX':
                if qty > sensex_max_qty:
                    print(f'{qty} is greater than allowed lot size qty of finifty {sensex_max_qty}')
                    qty=sensex_max_qty
            
            elif individual_values[f'index_{i}']=='BANKEX':
                if qty > bankex_max_qty:
                    print(f'{qty} is greater than allowed lot size qty of finifty {bankex_max_qty}')
                    qty= bankex_max_qty

            elif individual_values[f'index_{i}']=='MIDCPNIFTY':
                if qty > midcpnifty_max_qty:
                    print(f'{qty} is greater than allowed lot size qty of finifty {midcpnifty_max_qty}')
                    qty=midcpnifty_max_qty
                        
            print('Identified Qty:',qty);logFile.flush(); os.fsync(logFile.fileno())
            logFile.flush(); os.fsync(logFile.fileno())
            return qty   
            
    def get_isl(i,df,logFile,order_exec_price):
        sl=last_2_candles_low_sl(df,logFile) 
        print('last two candles low:',sl)
        print('order exec price:', order_exec_price)
        if sl< (order_exec_price - (individual_values[f'isl_percentage_{i}'] *order_exec_price)):
            sl= order_exec_price - (individual_values[f'isl_percentage_{i}'] *order_exec_price)
        
        return sl
    
    def get_last_bearish_candle(i,df,logFile,order_exec_price):
        sl= calculate_last_bearish_candle(df, logFile)
        print('last bearish candle low is',sl)
        logFile.flush(); os.fsync(logFile.fileno())

        if sl< (order_exec_price - (individual_values[f'isl_percentage_{i}'] *order_exec_price)):
            sl= order_exec_price - (individual_values[f'isl_percentage_{i}'] *order_exec_price) 
        return sl

    def get_last_bearish_candle_with_rsi(i,df,logFile,order_exec_price,lower_limit_value):
        sl= calculate_last_bearish_candle_with_rsi(df, logFile,lower_limit_value)
        print('last bearish candle low is',sl)
        logFile.flush(); os.fsync(logFile.fileno())

        if sl< (order_exec_price - (individual_values[f'isl_percentage_{i}'] *order_exec_price)):
            sl= order_exec_price - (individual_values[f'isl_percentage_{i}'] *order_exec_price) 
        return sl


    
    
    def full_trailing_checking(trade,opt_low,isl_value,sl_order_id,logFile,all_exited,local_quantity):
        if opt_low > isl_value:
            sl_price= round(float(opt_low),1)
            price= round(float(opt_low-0.1),1) 
            sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
            
            

            isl_value=opt_low
            print("Shifting isl/tsl value")
            print("New tsl is",isl_value)
            logFile.flush(); os.fsync(logFile.fileno())
        return sl_order_id,isl_value,all_exited
    
    def check_tsl(prev_rsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity):
        
        if (((call_trade_taken and prev_rsi>rsi and (orsi < individual_values[f'upper_limit_{i}'] 
                )) or (put_trade_taken and rsi>prev_rsi
                and (orsi < individual_values[f'upper_limit_{i}'])))
                and (not entered_condition) 
                and ts1_condition_satisfied 
                and (individual_values[f'tsl1_{i}'][3:]== 'Exit at market price and half trailing with bearish candle low sl' or individual_values[f'tsl1_{i}'][3:]== 'Exit half and remaining at bearish candle low')) :
        
        
                                           
            print(f"given {individual_values[f'tsl1_{i}']} condition satisfied for the candle time of {last_candle_time}")
            logFile.flush(); os.fsync(logFile.fileno())

            entered_condition = True
            all_exited, sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,logFile,i,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  

        
        elif ( ((call_trade_taken and prev_rsi>rsi
                and  orsi< individual_values[f'lower_limit_{i}'])
                or (put_trade_taken and rsi>prev_rsi
                and  orsi< individual_values[f'lower_limit_{i}']))
                and (not entered_condition)  and (individual_values[f'tsl2_{i}'][3:]== 'Exit at market price and half trailing with bearish candle low sl' or individual_values[f'tsl2_{i}'][3:]== 'Exit half and remaining at bearish candle low')):
        # if True:  
                print(f"given {individual_values[f'tsl2_{i}']} condition satisfied for the candle time of {last_candle_time}")                                                        
                logFile.flush(); os.fsync(logFile.fileno())
                
                entered_condition = True  
                all_exited,sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,logFile,i,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  

        else:
            print("TSL condition not satisfies so not taking any trades")
            logFile.flush(); os.fsync(logFile.fileno())

        return all_exited,sl_order_id,local_quantity,entered_condition,isl_value

    def check_tsl_with_extra_condition(prev_rsi,trade,rsi,last_candle_time,opt_low,isl_value,logFile,order_exec_price,opt_tf,i,name,index_future,call_trade_taken,put_trade_taken,o,c,orsi,entered_condition,ts1_condition_satisfied,ts2_condition_satisfied,all_exited, sl_order_id,local_quantity):
        
        if (((call_trade_taken and prev_rsi>rsi and orsi < individual_values[f'upper_limit_{i}']) 
                or (put_trade_taken and rsi>prev_rsi and orsi < individual_values[f'upper_limit_{i}'])) 
                and not entered_condition
                and ts1_condition_satisfied 
                and individual_values[f'tsl1_{i}'][3:]== 'Full trailing and half trailing with bearish candle low sl') :
                                           
            print(f"given {individual_values[f'tsl1_{i}']} condition satisfied for the candle time of {last_candle_time}")                                                        
            logFile.flush(); os.fsync(logFile.fileno())
            
            entered_condition = True
            all_exited, sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,logFile,i,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  
        
        elif (((call_trade_taken and prev_rsi>rsi  and  orsi< individual_values[f'lower_limit_{i}'])
                or (put_trade_taken and rsi>prev_rsi  and  orsi< individual_values[f'lower_limit_{i}']))
                and not entered_condition  
                and ts2_condition_satisfied 
                and individual_values[f'tsl2_{i}'][3:]== 'Full trailing and half trailing with bearish candle low sl' ):
        # if True:  
                print(f"given {individual_values[f'tsl2_{i}']} condition satisfied for the candle time of {last_candle_time}")                                                        
                logFile.flush(); os.fsync(logFile.fileno())
                
                entered_condition = True  
                all_exited,sl_order_id,local_quantity,isl_value=logic_checking(opt_low,trade,isl_value,logFile,i,sl_order_id,order_exec_price,name,opt_tf,local_quantity,all_exited)  

        
        elif (opt_low > isl_value and not entered_condition):
                sl_price= round(float(opt_low),1)
                price= round(float(opt_low-0.1),1) 
                sl_order_id,all_exited= orderplacement.modify_sl_order(sl_order_id,sl_price,price,local_quantity,all_exited)
                

                isl_value=opt_low
                print("isl value is",isl_value)
                logFile.flush(); os.fsync(logFile.fileno())

        return all_exited, sl_order_id,local_quantity,entered_condition,isl_value


    def get_trade_signal_from_file(filename,logFile):
        trade_signals = []

        while True:
            try:
                with open(filename, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        trade_signals.append(line.strip())  
                break  

            except IOError as e:
                print(f"An error occurred while reading the file: {e}. Retrying in a few seconds.")
                time.sleep(2)
        print('trade signals is',trade_signals)
        return trade_signals
    
    def can_proceed_with_trade(desired_trade, filename,logFile):
        existing_trade = get_trade_signal_from_file(filename,logFile)
        print('existing trade is',existing_trade)
        print('desired trade is',desired_trade)
        if not existing_trade:
            return True
        elif desired_trade == "put" and "call" in existing_trade:
            return False
        elif desired_trade == "call" and "put" in existing_trade:
            return False
        else:
            return True
        
    def append_to_file(filename, content, logFile, max_retries=5):
        retry_count = 0
        
        while retry_count < max_retries:
            with file_lock:  # Acquiring the lock to ensure thread-safe file operations
                try:
                    with open(filename, "a") as file:  
                        file.write(content + "\n") 
                        print(f'{content} got written')
                        break  # Exit the loop if operation is successful
                    
                except Exception as e:  # Catching general exceptions for robust error handling
                    print(f"An error occurred while appending to the file: {e}. Retrying...")
                    retry_count += 1  # Increment the retry count
                    time.sleep(2)  # Wait for 2 seconds before retrying
                    
        if retry_count == max_retries:
            print("Max retry attempts reached. Could not append to the file.")


    

    ################ ENTRY CONDITION LOGIC FUNCTION   ###############
    def create_trade_conditions(candle_condition, previous_candle, close_condition, condition, 
                        call_candle_condition, call_previous_candle, call_close_condition, call_condition,
                        put_candle_condition, put_previous_candle, 
                        put_close_condition, put_condition,index_future,pe_tradingsymbol,ce_tradingsymbol,logFile, i, bnf_tf,opt_tf,listcheck):
            
            print("entered create trade")
            logFile.flush(); os.fsync(logFile.fileno())
            while True:
                now= datetime.datetime.now()
                print('time is', now)                
                if 4 < now.second < 50:
                    break
                time.sleep(1)
            C1,C2,O2,H2,L2,RSI,prev_rsi,trade_taken_time,CC1,CC2,CO2,CH2,CL2,CRSI,prev_crsi,callp1_time,PC1,PC2,PO2,PH2,PL2,PRSI,prev_prsi,putp1_time,cdf,pdf,supertrend_value,call_supertrend_value,put_supertrend_value,ce_rsi_df,pe_rsi_df,fut_rsi_df= data_analysis(i,index_future,pe_tradingsymbol,ce_tradingsymbol,logFile, bnf_tf,opt_tf)
            #C1,C2,O2,H2,L2,RSI,prev_rsi,CC1,CC2,CO2,CH2,CL2,CRSI,prev_crsi,PC1,PC2,PO2,PH2,PL2,PRSI,prev_prsi,cdf,pdf,trade_taken_time
    
            UL=individual_values[f'upper_limit_{i}']
            LL=individual_values[f'lower_limit_{i}']
            new_LL= individual_values[f'put_ll_{i}']
            new_UL= individual_values[f'put_ul_{i}']
            print(new_LL,new_UL)
    
            print('Returning all the values time:',datetime.datetime.now())
            logFile.flush(); os.fsync(logFile.fileno())
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
            
            
            print("Third chart reference is",individual_values[f'third_chart_reference_{i}'])
            print("option chart or future chart is",individual_values[f'chart_reference_{i}'])


            if individual_values[f'chart_reference_{i}']== 'future_chart':
                print("Selected Future Chart")
                if individual_values[f'third_chart_reference_{i}']== 'yes':


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

                elif individual_values[f'third_chart_reference_{i}']== 'no':

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

            elif individual_values[f'chart_reference_{i}']== 'option_chart':
                print("Selected Option Chart")


                if individual_values[f'third_chart_reference_{i}']== 'yes':


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

                elif individual_values[f'third_chart_reference_{i}']== 'no':

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
            print('\nCondition checking!:',print_str,'UL-->:',individual_values[f'upper_limit_{i}'],'LL-->:',individual_values[f'lower_limit_{i}'])
            print('CALL entry Conditions:',(call_condition_str),eval(call_condition_str))
            print('PUT entry Conditions:',(put_condition_str),eval(put_condition_str))
            
                    
            #exit()
            #print('Time after eval function',datetime.datetime.now() )
            logFile.flush(); os.fsync(logFile.fileno())


            if individual_values[f'chart_reference_{i}']== 'future_chart':
    
                if individual_values[f'buy_only_call_or_put_{i}']=="both":     
                    #time.sleep(60)
                    if individual_values[f'supertrend_{i}']=='yes':
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
                        logFile.flush(); os.fsync(logFile.fileno())
                        signal=True
                        if signal:
                            t1= cdf['date'].iloc[-1]
                            t2= cdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                print('t1 and t2 not in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                                logFile.flush(); os.fsync(logFile.fileno())

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

                                logFile.flush(); os.fsync(logFile.fileno())



                        

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
                                logFile.flush(); os.fsync(logFile.fileno())
                            else:
                                call_condition_met=None
                                put_condition_met=None 
                                print("trade had benn already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                            
                    logFile.flush(); os.fsync(logFile.fileno())
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df
                    
            
                elif individual_values[f'buy_only_call_or_put_{i}']=="call":
                    if individual_values[f'supertrend_{i}']=='yes':
                        
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
                                logFile.flush(); os.fsync(logFile.fileno())
                            else:
                                call_condition_met=None
                                put_condition_met=None
                                print("trade had been already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                                logFile.flush(); os.fsync(logFile.fileno())


                            
                    
                    logFile.flush(); os.fsync(logFile.fileno())
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df
                    
        
                elif individual_values[f'buy_only_call_or_put_{i}']=="put":
                    if individual_values[f'supertrend_{i}']=='yes':
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
                                logFile.flush(); os.fsync(logFile.fileno())
                            else:
                                call_condition_met=None
                                put_condition_met=None 
                                print("trade had been already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                
                    logFile.flush(); os.fsync(logFile.fileno())
                    print(call_condition_met,put_condition_met)
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df


            elif individual_values[f'chart_reference_{i}']== 'option_chart':
                if individual_values[f'buy_only_call_or_put_{i}']=="both":     
                    #time.sleep(60)
                    if individual_values[f'supertrend_{i}']=='yes':
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
                        logFile.flush(); os.fsync(logFile.fileno())
                        signal=True
                        if signal:
                            t1= cdf['date'].iloc[-1]
                            t2= cdf['date'].iloc[-2]
                            if not (t1 in listcheck and t2 in listcheck):
                                print('t1 and t2 not in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                                logFile.flush(); os.fsync(logFile.fileno())

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

                                logFile.flush(); os.fsync(logFile.fileno())



                        

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
                                logFile.flush(); os.fsync(logFile.fileno())
                            else:
                                call_condition_met=None
                                put_condition_met=None 
                                print("trade had benn already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                            
                    logFile.flush(); os.fsync(logFile.fileno())
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df
                    
            
                elif individual_values[f'buy_only_call_or_put_{i}']=="call":
                    if individual_values[f'supertrend_{i}']=='yes':
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
                                logFile.flush(); os.fsync(logFile.fileno())
                            else:
                                call_condition_met=None
                                put_condition_met=None
                                print("trade had been already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                                logFile.flush(); os.fsync(logFile.fileno())


                            
                    
                    logFile.flush(); os.fsync(logFile.fileno())
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df
                    
        
                elif individual_values[f'buy_only_call_or_put_{i}']=="put":
                    print("call st value is",call_supertrend_value)
                    print("put st value is",put_supertrend_value)
                    if individual_values[f'supertrend_{i}']=='yes':
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
                                logFile.flush(); os.fsync(logFile.fileno())
                            else:
                                call_condition_met=None
                                put_condition_met=None 
                                print("trade had been already taken for the entry candle so not taking trade again")
                                print('t1 and t2 is already in listcheck')
                                print('t1 is',t1, 't2 is',t2, 'listcheck is',listcheck)
                
                    logFile.flush(); os.fsync(logFile.fileno())
                    print(call_condition_met,put_condition_met)
                    return call_condition_met,put_condition_met,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df




    def full_logic(i,logFile,index_future,pe_tradingsymbol,ce_tradingsymbol,premium,time_frame_dict,listcheck):
            global bnf_tf1,opt_tf1
            user_opt_tf=individual_values[f'call_time_frame_{i}']
            user_bnf_tf=individual_values[f'bnf_time_frame_{i}']
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
            
            
           
            logFile.flush(); os.fsync(logFile.fileno())
         
            call_trade,put_trade,cdf,pdf,listcheck,supertrend_value,CC2,CO2,CH2,CL2,PC2,PO2,PH2,PL2,ce_rsi_df,pe_rsi_df,fut_rsi_df = create_trade_conditions(*selected_values,logFile, i, bnf_tf,opt_tf,listcheck)
            print('new',call_trade,put_trade,supertrend_value)
            logFile.flush(); os.fsync(logFile.fileno())
            
            if call_trade:
                        print("Given conditions:",selected_values)
                        
                        trade="call"
                        name= ce_tradingsymbol
                        ltp= get_ltp(name,logFile)
                        print("ltp is", ltp)
                        print("permissible lower limit is", individual_values[f'premium_limit_{i}'])
                        if  individual_values[f'premium_limit_{i}'] > ltp:
                            allowed=individual_values[f'premium_limit_{i}']
                            print(f'ltp {ltp} is lesser than permissible premium range {allowed} so not proceeding with trade. Scanning starting again')

                            call_trade=False
                        else:
                            allowed=individual_values[f'premium_limit_{i}']
                            print(f'ltp {ltp} is greater than permissible premium range {allowed} so proceeding with trade')

                        #opt_tf=individual_values[f'call_time_frame_{i}']
            if put_trade:
                        print("Given conditions:",selected_values)
                        
                        trade="put"
                        name= pe_tradingsymbol
                        ltp= get_ltp(name,logFile)
                        print("ltp is", ltp)
                        print("permissible lower limit is", individual_values[f'premium_limit_{i}'])
                        if  individual_values[f'premium_limit_{i}'] > ltp:
                            allowed=individual_values[f'premium_limit_{i}']
                            print(f'ltp {ltp} is lesser than permissible premium range {allowed} so not proceeding with trade. Scanning starting again')

                            put_trade=False
                        else:
                            allowed=individual_values[f'premium_limit_{i}']
                            print(f'ltp {ltp} is greater than permissible premium range {allowed} ')


                        #opt_tf=individual_values[f'put_time_frame_{i}']
                          
            logFile.flush(); os.fsync(logFile.fileno())
            trade_checking={}
            call_trade_taken=False
            put_trade_taken=False
            print("order type is",individual_values[f'order_type_{i}'] )


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


    def monitor_options_algo(premium,logFile,strategy_no):
        i=strategy_no
        print(f"Inputs for the following strategy ")
        print(f"""Entry time is {individual_values[f"entry_time_{i}"]}, Exit time is {individual_values[f"exit_time_{i}"]}, 
                        ENABLE Strategy is {individual_values[f"enable_strategy_{i}"]}
                        Buy only call or put is {individual_values[f"buy_only_call_or_put_{i}"]}, Dynamic stop is {individual_values[f"dynamic_stop_{i}"]}, Premium Range1 is {individual_values[f"premium_range1_{i}"]}
                        Premium Range2 is {individual_values[f"premium_range2_{i}"]}, Premium Range3 is {individual_values[f"premium_range3_{i}"]}, Premium Range4 is {individual_values[f"premium_range4_{i}"]}
                        CBT is {individual_values[f"cbt_{i}"]}, Capital to Deploy 
                        is {individual_values[f"capital_to_deploy_{i}"]}, QBT is {individual_values[f"qbt_{i}"]}
                        Quantity is {individual_values[f"quantity_{i}"]}, index is {individual_values[f"index_{i}"]}, Trade set up is {individual_values[f"trade_set_up_{i}"]}
                        RSI is {individual_values[f"rsi_{i}"]}, Upper Limit is {individual_values[f"upper_limit_{i}"]}, Lower Limit is {individual_values[f"lower_limit_{i}"]}
                        BNF Time Frame is {individual_values[f"bnf_time_frame_{i}"]}, Candle Condition is {individual_values[f"candle_condition_{i}"]}, Previous candle is {individual_values[f"previous_candle_{i}"]}
                        Close is {individual_values[f"close_{i}"]}, Condition is {individual_values[f"condition_{i}"]}, CALL Time Frame is {individual_values[f"call_time_frame_{i}"]}
                        Call Candle Condition is {individual_values[f"call_candle_condition_{i}"]}, Call previous candle is {individual_values[f"call_previous_candle_{i}"]}, Call close is {individual_values[f"call_close_{i}"]}
                        Call condition is {individual_values[f"call_condition_{i}"]}, Put Time frame is {individual_values[f"put_time_frame_{i}"]}, Put candle condition is {individual_values[f"put_candle_condition_{i}"]}
                        Put Previous Candle is {individual_values[f"put_previous_candle_{i}"]}, Put Close is {individual_values[f"put_close_{i}"]}, Put Condition is {individual_values[f"put_condition_{i}"]}
                        Order type is {individual_values[f"order_type_{i}"]}, ISL 
                        is {individual_values[f"isl_{i}"]}, Isl_percentage is {individual_values[f"isl_percentage_{i}"]}
                        TSL1 is {individual_values[f"tsl1_{i}"]}, TSL2 is {individual_values[f"tsl2_{i}"]}, Target is {individual_values[f"target_{i}"]}
                        Single candle condition is {individual_values[f"single_candle_condition_{i}"]}, TSL of SCC is {individual_values[f"tsl_of_scc_{i}"]}, After SCC X pct price Move is {individual_values[f"after_scc_x_pct_price_move_{i}"]}
                        After SCC Y pct Trailing Move is {individual_values[f"after_scc_y_pct_trailing_move_{i}"]}, bnf lot size is {individual_values[f"bnf_lot_size_{i}"]}, nifty lot size is {individual_values[f"nifty_lot_size_{i}"]}
                        finnifty lot size is {individual_values[f"finnifty_lot_size_{i}"]}, first target is {individual_values[f'first_target_{i}']}, second target is {individual_values[f'second_target_{i}']},
                        first target percent is {individual_values[f'first_target_trailing_{i}']},target split enabled or disabled is {individual_values[f'target_split_{i}']}, slm order value is {individual_values[f'limit_order_value_{i}']},
                        second target percent is {individual_values[f'second_target_trailing_{i}']} """)
        listcheck=[]
        while events_algo[premium].is_set():
            logFile.flush(); os.fsync(logFile.fileno())
            
            index= individual_values[f'index_{i}']
            print("index to Scan:",index)
            index_future,strike_price_list= get_strikeprice(index,premium)

            #index_future= 'BANKNIFTY23AUGFUT'
            #strike_price_list= ['BANKNIFTY2381743600CE','BANKNIFTY2381743600PE']

            print(index_future,strike_price_list)
            logFile.flush(); os.fsync(logFile.fileno())
    
            for tradingsymbol in strike_price_list:
                if tradingsymbol[-2:].upper() == 'PE':
                    pe_tradingsymbol = tradingsymbol
                elif tradingsymbol[-2:].upper() == 'CE':
                    ce_tradingsymbol = tradingsymbol
    
            print('Scanning options:',i,index_future,pe_tradingsymbol,ce_tradingsymbol,'premium:',premium)
            logFile.flush(); os.fsync(logFile.fileno())

            if index=='MIDCPNIFTY':
                index_future='NIFTY MID SELECT'
            elif index=='SENSEX':
                index_future='SENSEX'
            elif index=='BANKEX':
                index_future='BANKEX'



    
            listcheck=full_logic(i,logFile,index_future,pe_tradingsymbol,ce_tradingsymbol,premium,time_frame_dict,listcheck)
            
            individual_value=read_excel(logFile)
            df = pd.read_excel('Index_BUY_Algo_Configurations.xlsm', sheet_name='Credentials', header=None)
            data_dict = dict(zip(df[0], df[1]))
            main_stop=data_dict['ALGO MODE']


            print('Dynamic stop Flag:',individual_value[f'dynamic_stop_{i}'])
            print("Main stop is",main_stop)
            if individual_value[f'dynamic_stop_{i}']=='yes' or main_stop=='Stop':
                events_algo[premium].clear()
                while True:
                    individual_value=read_excel(logFile)
                    df = pd.read_excel('Index_BUY_Algo_Configurations.xlsm', sheet_name='Credentials', header=None)                            
                    data_dict = dict(zip(df[0], df[1]))
                    main_stop=data_dict['ALGO MODE']
                    print("main stop is", main_stop)
                    print(individual_value[f'dynamic_stop_{i}'])
                    if individual_value[f'dynamic_stop_{i}']=='no'and main_stop=='Start':
                        events_algo[premium].set()                        
                        break
                    time.sleep(2)
            
            current_time= datetime.datetime.now().time()
            exit_time= individual_values[f'exit_time_{i}']                       
            if current_time>=exit_time:
                events_algo[premium].clear()
                print('\n\nEOD Exit time reached. So exiting the ALGO!!')
                logFile.flush(); os.fsync(logFile.fileno())
                exit()
            
    
    
    # for premium in strike_prices_algo: 
    #     print('premium',premium)
    #     print(strike_prices_algo)
    #     logFile.flush(); os.fsync(logFile.fileno())

    #     monitor_options_algo(premium,logFile,strategy_no,strike_no)

    for strike_price in strike_prices_algo:
        print('Premium Bucket:',strike_price)
        monitor_options_algo(strike_price,logFile,strategy_no)

   else:
    print('This Bucket Strategy & Premium bucket:',strategy_no,premium_bucket+1,'. Premium bucket is set as Not Applicable. So no scanning for this.!! Good Bye!')
    logFile.flush(); os.fsync(logFile.fileno())
    exit()
  else:
    print('Strategy Disabled: This Bucket Strategy & Premium bucket:',strategy_no,premium_bucket+1,'is not enabled. So no scanning for this.!! Good Bye!')
    logFile.flush(); os.fsync(logFile.fileno())
    exit()

    ## hardcoding needs to correct.

    #after exit time e.g1530 algo should square off all the open positons and it shoould exit. -- priority 1
    #start of the algo all the strategy inputs needs to be displayed on log. -- priority 1
    #signal start/stop button to start/stop the batches.. right now looks each batch has its own.. os main start/stop needed on the config sheet. -- priority 1
    #need to get the available maring from zerodha and need to allocate funds based on given % utilization. -- priority 2
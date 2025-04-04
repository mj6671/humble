import pandas as pd
import pandas_ta as ta

def calculate_supertrend(df, period=10, multiplier=3):
    """
    Calculates the Supertrend indicator and adds it to the DataFrame.
    
    Parameters:
    - df (DataFrame): A DataFrame containing 'high', 'low', and 'close' price columns.
    - period (int): The ATR period to calculate Supertrend.
    - multiplier (float): The multiplier for ATR to calculate bands.
    
    Returns:
    - DataFrame: The original DataFrame with a new 'supertrend' column.
    """
    # Check if required columns exist
    if not {'high', 'low', 'close'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
    
    # Calculate Average True Range (ATR)
    atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=period)
    if atr.isnull().all():
        raise ValueError("ATR calculation failed. Ensure your data and parameters are correct.")

    # Calculate hl2 (high-low midpoint)
    hl2 = (df['high'] + df['low']) / 2

    # Calculate upper and lower bands
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    # Initialize Supertrend column with NaN
    supertrend = pd.Series(index=df.index, dtype='float64')

    # Set initial Supertrend value based on close price
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = hl2.iloc[i]  # Use hl2 for the first row
        else:
            if df['close'].iloc[i] > upperband.iloc[i - 1]:
                supertrend.iloc[i] = lowerband.iloc[i]
            elif df['close'].iloc[i] < lowerband.iloc[i - 1]:
                supertrend.iloc[i] = upperband.iloc[i]
            else:
                supertrend.iloc[i] = supertrend.iloc[i - 1]

    # Add Supertrend to the DataFrame
    df['supertrend'] = supertrend

    return df


def check_candle_condition(df, candle_condition):
    """Checks the candle condition."""
    if candle_condition['type'] == 'green':
        return (df['close'] >= df['open']) if candle_condition['comparison'] == '>=' else (df['close'] > df['open'])
    elif candle_condition['type'] == 'red':
        return (df['close'] <= df['open']) if candle_condition['comparison'] == '<=' else (df['close'] < df['open'])

def check_rsi_condition(df, rsi_condition, rsi_length=14, rsi_limits=(30, 50, 70)):
    """Checks the RSI condition."""
    df['rsi'] = ta.rsi(df['close'], length=rsi_length)
    result = pd.Series(False, index=df.index)

    if rsi_condition['comparison'] == 'higher':
        result = df['rsi'] > df['rsi'].shift(1)
    elif rsi_condition['comparison'] == 'lower':
        result = df['rsi'] < df['rsi'].shift(1)

    lower, middle, upper = rsi_limits
    if rsi_condition['limits'] == 'below':
        result &= df['rsi'] < lower
    elif rsi_condition['limits'] == 'middle':
        result &= (df['rsi'] >= lower) & (df['rsi'] <= upper)
    elif rsi_condition['limits'] == 'above':
        result &= df['rsi'] > upper

    return result

def check_supertrend_condition(df, supertrend_condition, supertrend_settings=(10, 3)):
    """Checks the Supertrend condition."""
    df = calculate_supertrend(df, period=supertrend_settings[0], multiplier=supertrend_settings[1])
    return df['supertrend'] == supertrend_condition

def check_entry_conditions(df, candle_condition, rsi_condition, supertrend_condition, trade_type,
                            rsi_length=14, rsi_limits=(30, 50, 70), supertrend_settings=(10, 3)):
    """
    Checks entry conditions based on candle, RSI, and Supertrend, and determines buy/sell signals.
    
    Parameters:
    df (DataFrame): OHLCV data.
    candle_condition (dict): {'type': 'green'/'red', 'comparison': '>=', 'value': float}
    rsi_condition (dict): {'comparison': 'higher'/'lower', 'limits': ('below', 'middle', 'above')}.
    supertrend_condition (str): 'green'/'red'.
    trade_type (str): 'buy'/'sell'.
    rsi_length (int): RSI period.
    rsi_limits (tuple): RSI thresholds (lower, middle, upper).
    supertrend_settings (tuple): Supertrend settings (period, multiplier).

    Returns:
    DataFrame: DataFrame with conditions and trade signals.
    """
    # Check individual conditions
    df['candle_condition'] = check_candle_condition(df, candle_condition)
    df['rsi_condition'] = check_rsi_condition(df, rsi_condition, rsi_length, rsi_limits)
    df['supertrend_condition'] = check_supertrend_condition(df, supertrend_condition, supertrend_settings)

    # Combine conditions based on trade type
    if trade_type == 'buy':
        df['entry_signal'] = (
            (df['candle_condition'] & df['rsi_condition'] & (df['supertrend_condition'] == 'green'))
        )
    elif trade_type == 'sell':
        df['entry_signal'] = (
            (df['candle_condition'] & df['rsi_condition'] & (df['supertrend_condition'] == 'red'))
        )

    return df
    
def main():
    # Example OHLCV DataFrame (replace with your real data source)
    data = {
        'open': [100, 102, 101, 103, 104],
        'high': [102, 104, 103, 105, 106],
        'low': [99, 101, 100, 102, 103],
        'close': [101, 103, 102, 104, 105],
        'volume': [1000, 1200, 1100, 1300, 1250]
    }
    df = pd.DataFrame(data)

    # Define conditions
    candle_condition = {'type': 'green', 'comparison': '>='}
    rsi_condition = {'comparison': 'higher', 'limits': 'middle'}
    supertrend_condition = 'green'

    # Call the entry condition function
    result_df = check_entry_conditions(
        df,
        candle_condition=candle_condition,
        rsi_condition=rsi_condition,
        supertrend_condition=supertrend_condition,
        trade_type='buy',
        rsi_length=14,
        rsi_limits=(30, 50, 70),
        supertrend_settings=(10, 3)
    )

    # Display the results
    print(result_df)

if __name__ == "__main__":
    main()

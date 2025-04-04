from config.database import db_connect 

# Fetch pending orders
def fetch_pending_orders():
    db = db_connect()
    cursor = db.cursor(dictionary=True)
    query = """
    SELECT uo.*, ua.status as uapi_status, el.status as exchange_status
    FROM trades as uo
    LEFT JOIN exchange_list as el ON el.exchange_id = uo.exchange_id 
    LEFT JOIN users_api as ua ON uo.user_id = ua.user_id AND uo.exchange_id = ua.exchange_id
    WHERE uo.status = 'Pending'  AND ua.status = 1 AND el.status = 1
    """
    cursor.execute(query)
    orders = cursor.fetchall()
    db.close()
    return orders

# Update users orders
def update_users_orders(bin_order, user_id, user_order_id, exchange_id):
    if bin_order['status'] == "FILLED":
        db = db_connect()
        cursor = db.cursor(dictionary=True)
        update_query = """
        UPDATE trades 
        SET status = 'Complete' 
        WHERE user_id = %s AND order_id = %s AND exchange_id = %s AND status = 'Pending'
        """
        cursor.execute(update_query, (user_id, user_order_id, exchange_id,))  # ✅ Fixed tuple issue
        db.commit()
        cursor.close()
        db.close()

# Update order status in database
def update_order_status(user_id, order_id, status):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE trades SET status=%s WHERE id=%s AND user_id=%s", (status, order_id, user_id))
    conn.commit()
    conn.close()
    
# Update trailing stoploss in database
def update_trailing_stoploss(user_id, order_id, trailing_stoploss):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE trades SET trailing_stoploss=%s WHERE id=%s AND user_id=%s", (trailing_stoploss, order_id, user_id))
    conn.commit()
    conn.close()
    
    
# Fetch user API keys
def fetch_user_api_keys(user_id, exchange_id):
    db = db_connect()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT api_key, secret_key FROM users_api WHERE user_id = %s AND exchange_id = %s AND status = '1'", 
                   (user_id, exchange_id))  
    user = cursor.fetchone()
    db.close()
    return user

def get_exchange_details(exchange_id):
    db = db_connect()
    cursor = db.cursor(dictionary=True)
    query = """SELECT * FROM `exchange_list` WHERE `exchange_id` = %s"""
    cursor.execute(query, (exchange_id,))  # ✅ Fixed tuple issue
    details = cursor.fetchone()
    db.close()
    return details

# Save Exchange order data
def save_order_to_db(bin_order, user_id, user_order_id, exchange_id, entry_conditions):
    db = db_connect()
    cursor = db.cursor()
    sql = """INSERT INTO exchange_orders 
            (users_order_id, user_id, exchange_id, exchange_order_id, symbol, side, price, quantity, status, entry_conditions) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = (
        user_order_id, user_id, exchange_id, bin_order['orderId'], bin_order['symbol'], bin_order['side'],
        bin_order['fills'][0]['price'], bin_order['executedQty'], bin_order['status'], entry_conditions or ""  # ✅ Ensuring entry_conditions is a valid string
    )
    cursor.execute(sql, tuple(values))  # ✅ Ensuring it's a tuple
    db.commit()
    cursor.close()
    db.close()
	
# Log trade in database
def log_trade(user_id, order_id, exchange_id, exchange_order_id, symbol, order_type, price, amount, status):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO order_book (user_id, order_id, exchange_id, exchange_order_id, currency, type, price, qty, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (user_id, order_id, exchange_id, exchange_order_id, symbol, order_type, price, amount, status)
    )
    conn.commit()
    conn.close()

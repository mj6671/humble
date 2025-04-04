from database import db_connect

# Fetch pending orders
def fetch_pending_orders():
    db = db_connect()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users_orders WHERE status = 'Pending'")
    orders = cursor.fetchall()
    db.close()
    return orders

# update users orders
def update_users_orders(bin_order, user_id, user_order_id):
    if bin_order['status'] == "FILLED":
        db = db_connect()
        cursor = db.cursor(dictionary=True)
        
        update_query = """
        UPDATE users_orders 
        SET status = 'Completed' 
        WHERE user_id = %s AND order_id = %s AND status = 'Pending'
        """
        
        cursor.execute(update_query, (user_id, user_order_id))
        db.commit()  # Save changes to the database
        
        cursor.close()
        db.close()

# Update order status in database
def update_order_status(user_id, order_id, status):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE users_orders SET status=%s WHERE order_id=%s AND user_id=%s", (status, order_id, user_id))
    conn.commit()
    conn.close()
# Fetch user API keys
def fetch_user_api_keys(user_id):
    db = db_connect()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT api_key, secret_key FROM users_api WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    db.close()
    return user

# Save Binance order data
def save_order_to_db(bin_order, user_id, user_order_id, entry_conditions):
    db = db_connect()
    cursor = db.cursor()
    sql = """INSERT INTO binance_orders 
            (users_order_id, user_id, binance_order_id, symbol, side, price, quantity, status, entry_conditions) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = (
        user_order_id, user_id, bin_order['orderId'], bin_order['symbol'], bin_order['side'],
        bin_order['fills'][0]['price'], bin_order['executedQty'], bin_order['status'], entry_conditions
    )
    cursor.execute(sql, values)
    db.commit()
    db.close()
	

# Log trade in database
def log_trade(user_id, order_id, symbol, order_type, price, amount, status):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO trade_logs (user_id, order_id, symbol, order_type, price, amount, status) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user_id, order_id, symbol, order_type, price, amount, status)
    )
    conn.commit()
    conn.close()

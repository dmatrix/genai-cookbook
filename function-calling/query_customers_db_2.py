import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('customers.db')
cursor = conn.cursor()

# Query for product with the highest price
cursor.execute('''
SELECT product_name, MAX(price_paid) FROM customer_data
''')
highest_price_product = cursor.fetchone()
print(f"Product with the highest price: {highest_price_product[0]}, Price: {highest_price_product[1]}")

# Query for product with the lowest price
cursor.execute('''
SELECT product_name, MIN(price_paid) FROM customer_data
''')
lowest_price_product = cursor.fetchone()
print(f"Product with the lowest price: {lowest_price_product[0]}, Price: {lowest_price_product[1]}")



# Close the database connection
conn.close()

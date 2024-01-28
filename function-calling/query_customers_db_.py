import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('customers.db')
cursor = conn.cursor()

# Query for product with the highest price and customer details
cursor.execute('''
SELECT product_name, customer_name, city, price_paid FROM customer_data
ORDER BY price_paid DESC LIMIT 1
''')
highest_price_details = cursor.fetchone()
print(f"Product with the highest price: {highest_price_details[0]}, Customer: {highest_price_details[1]}, City: {highest_price_details[2]}, Price: {highest_price_details[3]}")

# Query for product with the lowest price and customer details
cursor.execute('''
SELECT product_name, customer_name, city, price_paid FROM customer_data
ORDER BY price_paid ASC LIMIT 1
''')
lowest_price_details = cursor.fetchone()
print(f"Product with the lowest price: {lowest_price_details[0]}, Customer: {lowest_price_details[1]}, City: {lowest_price_details[2]}, Price: {lowest_price_details[3]}")

# List of all customers who bought the product with the highest price
highest_price_product = highest_price_details[0]
cursor.execute('''
SELECT DISTINCT customer_name FROM customer_data
WHERE product_name = ? AND price_paid = ?
''', (highest_price_product, highest_price_details[3]))
customers_highest_price_product = cursor.fetchall()
print(f"Customers who bought the product with the highest price ({highest_price_product}):")
for customer in customers_highest_price_product:
    print(customer[0])

# Close the database connection
conn.close()

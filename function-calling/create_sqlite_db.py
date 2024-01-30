import sqlite3
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Create a connection to the SQLite database
conn = sqlite3.connect('customers.db')
cursor = conn.cursor()

# Create the table
cursor.execute('''
CREATE TABLE IF NOT EXISTS customer_data (
    customer_name TEXT,
    product_name TEXT,
    date_of_purchase DATE,
    date_joined DATE,
    price_paid REAL,
    city TEXT
)
''')

# Sample data for product names
products = ['Laptop', 'Smartphone', 'Headphones', 
            'Camera', 'Smartwatch', 'Backpack', 
            'Sunglasses', 'Book', 'Shoes', 
            'T-shirt', 'Pants', 'Shorts',
            'Skirt', 'Dress', 'Sweater']

# Generate and insert the data
for _ in range(500_000):
    customer_name = fake.name()
    product_name = random.choice(products)
    date_of_purchase = fake.date()
    date_joined = fake.date()
    price_paid = round(random.uniform(20, 1000), 2)
    city = fake.city()

    cursor.execute('''
    INSERT INTO customer_data (customer_name, product_name, date_of_purchase, date_joined, price_paid, city) 
    VALUES (?, ?, ?, ?, ? ,?)
    ''', (customer_name, product_name, date_of_purchase, date_joined, price_paid, city))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Data generation complete!")

import json
import random
from pathlib import Path

# Product categories and their descriptions
product_categories = {
    'Shoes': 'Comfortable and stylish footwear for all occasions.',
    'Books': 'A range of books from bestsellers to literary classics.',
    'Phones': 'Latest smartphones with cutting-edge technology.',
    'Laptops': 'High-performance laptops for work and play.',
    'Jackets': 'Fashionable jackets for all seasons.',
    'Pants': 'Comfortable and durable pants for everyday wear.',
    'Kitchen Utensils': 'Essential utensils for your cooking needs.',
    'Living Room Furniture': 'Elegant and cozy furniture for your living room.',
    'Home Improvement Tools': 'Reliable tools for your home repair and improvement projects.'
}

# Possible locations
locations = ['San Francisco', 'New York', 'Los Angeles', 'Seattle', 'San Diego', 'Chicago', 'Boston', 'Dallas', 'Houston']

# Generate 1M product definitions
product_definitions = []
NUM_PRODUCTS = [1_000, 10_000, 100_000, 1_000_000]

for np in NUM_PRODUCTS:
    for i in range(np):  
        category = random.choice(list(product_categories.keys()))
        product_def = {
            'category': category,
            'description': product_categories[category],
            'price': round(random.uniform(10, 1000), 2),  # Price between 10 and 1000
            'number_purchased': random.randint(1, 6),   # Number purchased between 1 and 100
            'consumer_rating': random.randint(1, 5),      # Rating between 1 and 5
            'gender': random.choice(['M', 'F']),
            'location': random.choice(locations)
        }
        product_definitions.append(product_def)
    file_name = "docs/product_definitions_" + str(np) + ".json"

    # Write the JSON data to a file
    with open(file_name, 'w') as file:
        json.dump(product_definitions, file, indent=4)
    print(f"JSON file created {np} product entries.")

# # Lets read the file and print the first 10 entries
# with open(file_name, 'r') as file:
#     product_definitions_read = json.load(file)
# print("First 5 entries:")
# for product_def in product_definitions_read[:5]:
#     print(product_def)

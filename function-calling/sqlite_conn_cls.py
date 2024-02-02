import sqlite3

class SQLiteDBSingleton:
    _instance = None
    _connection = None
    _cursor = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SQLiteDBSingleton, cls).__new__(cls)
        return cls._instance

    def create(self, db_path):
        if self._connection is None:
            self._connection = sqlite3.connect(db_path)
            print(f"SQLite Database connected to {db_path}")
        return self._connection

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            print("SQLite Database connection closed.")

    @property
    def connection(self):
        if self._connection is None:
            raise ValueError("Connection is not created yet. Use the create method to create a connection.")
        return self._connection
    @property
    def cursor(self):
        if self._cursor is None:
            self._cursor = self._connection.cursor()
        return self._cursor

if __name__ == "__main__":
    # Usage example
    db_singleton = SQLiteDBSingleton()
    db_connection = db_singleton.create('example.db')  # Creates the connection
    print(db_singleton.connection)  # Access the connection

    # Access the connection
    conn = db_singleton.connection
    cursor= db_singleton.cursor

    # Perform database operations using conn ...
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

    # List all tables in the database
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(tables.fetchall())

    # list all columns in the customer_data table
    print(cursor.execute("PRAGMA table_info('customer_data');").fetchall())


    # Close the connection when done
    db_singleton.close()

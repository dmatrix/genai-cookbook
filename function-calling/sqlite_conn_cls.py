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
    
    def execute(self, query) -> list[tuple]:
        
        """Return a list of tuples containing the query results."""
        try:
            results = self._connection.execute(query).fetchall()
        except Exception as e:
            results = f"query failed with error: {e}"
        return results
            
    def get_table_names(self) -> list[str]:
        """Return a list of table names."""
        table_names = []
        tables = self.execute("SELECT name FROM sqlite_master WHERE type='table';")
        for table in tables:
            table_names.append(table[0])
        return table_names
    
    def get_column_names(self, table_name:str) -> list[str]:
        """Return a list of column names."""
        column_names = []
        columns = self.execute(f"PRAGMA table_info('{table_name}');")
        for col in columns:
            column_names.append(col[1])
        return column_names
    
    def get_all_city_names(self) -> list[str]:
        """Return a list of all city names."""
        city_names = []
        cities = self.execute("SELECT DISTINCT city FROM customer_data;")
        for city in cities:
            city_names.append(city[0])
        return city_names
        
    def get_database_info(self) -> list[dict[str, list[str]]]:
        """Return a list of dicts containing the table name and columns for each table in the database."""
        table_dicts = []
        for table_name in self.get_table_names():
            columns_names = self.get_column_names(table_name)
            table_dicts.append({"table_name": table_name, "column_names": columns_names})
        return table_dicts
    
    def get_database_schema(self) -> str:
        database_schema_dict = self.get_database_info()
        database_schema_string = "\n".join(
            [
                f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
            for table in database_schema_dict
            ]
        )
        return database_schema_string

if __name__ == "__main__":
    # Usage example
    db_singleton = SQLiteDBSingleton()
    db_connection = db_singleton.create('customers.db')  # Creates the connection
    print(db_singleton.connection)  # Access the connection
    # Perform database operations using conn ...
    # Create the table
    db_singleton.execute('''
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
    tables = db_singleton.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(tables)

    # list all columns in the customer_data table
    print(db_singleton.get_table_names())

    # print all column names in the customer_data table
    print(db_singleton.get_column_names('customer_data'))

    # get all city names
    print(db_singleton.get_all_city_names())

    # get databse info
    print(db_singleton.get_database_info())

    # get database schema
    print(db_singleton.get_database_schema())

    # Close the connection when done
    db_singleton.close()

import sqlite3
import json
from sqlite_conn_cls import SQLiteDBSingleton

# Connect to the SQLite database
def connect_db(db_name) -> sqlite3.Connection:
    conn = sqlite3.connect(db_name)
    return conn

def get_table_names(conn:sqlite3.Connection) -> list[str]:
    """Return a list of table names."""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names

def get_all_city_names(conn:sqlite3.Connection) -> list[str]:
    """Return a list of all city names."""
    city_names = []
    cities = conn.execute("SELECT DISTINCT city FROM customer_data;")
    for city in cities.fetchall():
        city_names.append(city[0])
    return city_names

def get_column_names(conn:sqlite3.Connection, table_name:str) -> list[str]:
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names

def get_database_info(conn:sqlite3.Connection) -> list[dict[str, list[str]]]:
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts

def query_customer_database(conn:sqlite3.Connection, query:str) -> list[tuple]:
    """Return a list of tuples containing the query results."""
    try:
        results = conn.execute(query).fetchall()
    except Exception as e:
        results = f"query failed with error: {e}"
    return results


def execute_function_call(conn:sqlite3.Connection, message:object) -> list[tuple]:
    """
    Execute a function call and return the results.
    Note: this function is only meant to be used with the compatible
    OpenAI API."""
    if message.tool_calls[0].function.name == "query_customer_database":
        query = json.loads(message.tool_calls[0].function.arguments)["query"]
        results = query_customer_database(conn, query)
    else:
        results = f"Error: function {message.tool_calls[0].function.name} does not exist"
    return results


def get_database_schema(conn:sqlite3.Connection) -> str:
    database_schema_dict = get_database_info(conn)
    database_schema_string = "\n".join(
        [
            f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
        ]
    )
    return database_schema_string

if __name__ == "__main__":

    # Use the Singleton class to access the database
    db_singleton = SQLiteDBSingleton()
    conn = db_singleton.create('customers.db')
    print(db_singleton.get_database_info())
    print("---" * 10)
    print(db_singleton.get_column_names( "customer_data"))
    print("---" * 10)
    print(db_singleton.get_table_names())
    print("---" * 10)
    print(db_singleton.get_database_schema())
    print("---" * 10)
    print(db_singleton.get_all_city_names())
    db_singleton.close()

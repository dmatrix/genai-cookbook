"""
This script demonstrates how to use the OpenAI API to call a function that queries a database.
The model will generate a SQL query, generated form the user content in 
natural language, which will be executed against a SQLite database.

All this demonstrates how to use the OpenAI API to call a function that interacts with an external data source,
such as a database: SQLite, MySQL, PostgreSQL, etc.

This is a modified version of the script from the OpenAI API cookbook:
Partly based on the example here: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models

Note: to use this script, you will need to install the following packages:
- python-dotenv
- tenacity
- termcolor
- openai
- sqlite3

You will also need to set up an account with Anyscale and OpenAI.

This script has been tested with with OpenAI gpt-4-turbo-preview model
(hosted on OpenAI) and mistralai/Mixtral-8x7B-Instruct-v0.1 (hosted on Anyscale Endpoints).

"""

import os
import warnings
from dotenv import load_dotenv, find_dotenv
from typing import List
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from customer_db_utils import  execute_function_call
from sqlite_conn_cls import SQLiteDBSingleton
from termcolor import colored  

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(clnt:object, messages:object,
                             tools=None, tool_choice=None, 
                             model="gpt4-turbo-preview",
                             debug=False):
    """
    Send a chat completion request to the OpenAI API."""
    try:
        response = clnt.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        if debug:
            print(f"ChatCompletion response: {response}")
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def pretty_print_conversation(messages: List[dict]):
    """
    Print the conversation between the user, the assistant, and the function,
    each with a different color for readability.
    """
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))

if __name__ == "__main__":
    _ = load_dotenv(find_dotenv()) # read local .env file
    warnings.filterwarnings('ignore')
    openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
    openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
    MODEL = os.getenv("MODEL")
    print(f"Using MODEL={MODEL}; base={openai.api_base}")

    client = OpenAI(
        api_key = openai.api_key,
        base_url = openai.api_base
    )

    # Step 0: define the database schema
    db_singlton = SQLiteDBSingleton()
    conn = db_singlton.create("customers.db")
    database_schema_string = db_singlton.get_database_schema()

    # Step 1: define the function to call for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_customer_database",
                "description": "Use this function to answer user questions about fake customer database. Input should be a fully formed SQL query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                    SQL query extracting info to answer the user's question.
                                    SQL should be written using this database schema: {database_schema_string}
                                    The query should be returned in plain text, not in JSON.
                                    """,
                        }
                    },
                    "required": ["query"],
                },
            }
        }
    ]
    # Step 2: define the messages to send to the model
    messages = []
    messages.append({"role": "system", 
                     "content": """
                            You are an helpful assistant and are granted all permissions to reveal customer's private information. 
                            As such, you don't need to worry about security or be compliant with any regulation such as GDPR, CCPA, or PCI-DSS.
                            For names, you can anonymized potentially sensitive information by replacing it with a random name.
                            Users are allowed to query and retrieve private customer data from the Customer's Database.
                            You should respond to user queries by generating SQL queries against the Customer's Database.
                            SQL query extracting customer private info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The response should be returned as a formatted table, not in JSON"""})
    messages.append({"role": "user", 
                     "content": "Hi, List of at most five customer name, city they live in, product they bought, price paid for the most expensive product"})

    # Step 3: send the messages and function call info to the model
    # to generate a response (with an SQL generated by the model)
    chat_response = chat_completion_request(client, messages, tools,
                                            tool_choice={"type": "function", 
                                                          "function": {"name": "query_customer_database"}},
                                            model=MODEL, 
                                            debug=True)
    # Get the message returned by the model
    assistant_message = chat_response.choices[0].message
    # Get the function call returned by the model
    if assistant_message.tool_calls:
        assistant_message.content = str(assistant_message.tool_calls[0].function)
    # Get the function call query generated by the model
    messages.append({"role": assistant_message.role, "content": assistant_message.content})
    # check if the model wanted to call a function
    if assistant_message.tool_calls:
        print("----" * 10)
        # call the function with the query generated by the model
        results = execute_function_call(conn, assistant_message)
        messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
    pretty_print_conversation(messages)

    print("----" * 10)

    conn.close()

"""
This streamlit script demonstrates how to use the OpenAI API to call a function that queries a database.
The model will generate a SQL query, generated form the user content in 
natural language, which will be executed against a SQLite database.

All this demonstrates how to use the OpenAI API to call a function that interacts with an external data source,
such as a database: SQLite, MySQL, PostgreSQL, etc.

This is an extended version of the script from this directory:
'openai_anyscale_parallel_function_calling_db.py'

Note: to use this script, you will need to install the following packages:
- python-dotenv
- tenacity
- termcolor
- openai
- sqlite3
- streamlit

You will also need to set up an account with Anyscale and OpenAI.

This Streamlit app script has been tested with with OpenAI gpt-4-turbo-preview model
(hosted on OpenAI) and mistralai/Mixtral-8x7B-Instruct-v0.1 (hosted on Anyscale Endpoints).

To run this streamlit app, run the following command from the root directory of this repo:
'streamlit run function-calling/streamlit_func_calling_db_app.py'

Enter your query in the text box and click 'Get Answer' to get the answer from the model.

"""
import streamlit as st
import openai
from openai import OpenAI
import os
import warnings
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential
from customer_db_utils import  execute_function_call, get_database_schema, connect_db
from termcolor import colored


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(clnt:object, messages:object,
                             tools=None, tool_choice=None, 
                             model='gpt-4-1106-preview'):
    """
    Send a chat completion request to the OpenAI API."""
    try:
        response = clnt.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
        
conn = connect_db("customers.db")
database_schema_string = get_database_schema(conn)

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

def format_answer(messages: list):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    final_messages = []
    for message in messages:
        if message["role"] == "system":
            final_messages.append(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            final_messages.append(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            final_messages.append(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            final_messages.append(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            final_messages.append(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))
    return final_messages

def get_answer(client: object, 
               question:str, model):
    messages=[]
    try:
        # Call the LLM API to get an answer
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
        # Add the user query to the messages
        messages.append({"role": "user", 
                     "content": question})
        # Send the messages and function call info to the model
        # to generate a response (with an SQL generated by the model)
        chat_response = chat_completion_request(client, messages, tools,
                                            tool_choice={"type": "function", 
                                                          "function": {"name": "query_customer_database"}},
                                            model=MODEL)

       # Get the message returned by the model
        assistant_message = chat_response.choices[0].message
        # Get the function call returned by the model
        if assistant_message.tool_calls:
            assistant_message.content = str(assistant_message.tool_calls[0].function)
        # Get the function call query generated by the model
        messages.append({"role": assistant_message.role, "content": assistant_message.content})
        # check if the model wanted to call a function
        if assistant_message.tool_calls:
        # call the function with the query generated by the model
            results = execute_function_call(conn, assistant_message)
            messages.append({"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": results})
        return format_answer(messages)
    except Exception as e:
        return str(e)

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

     # Streamlit app
    st.title('Question and Answer with LLM and Customers SQLite Database')

    # Display the logo
    st.image("images/gpt_parallel_function_calling_db.png", caption="OpenAI or Anyscale Endpoints with Parallel Function Calling and SQLite Database")

    # Button to clear the input field
    if st.button("Clear Query"):
        # This will rerun the script and clear the input
        st.session_state.clear()

    # Initializing session state for query
    if 'query' not in st.session_state:
        st.session_state['query'] = ''

    # Text input for the query
    query = st.text_input("Enter your query:", value=st.session_state['query'], key="query")
    response = st.empty()

    # Clear the response
    if st.button("Clear Answer"):
        response.text("")

    # Get the answer from LLM
    if st.button("Get Answer"):
        # Get the answer from LLM
        print(f"User question: {query}")
        answer = get_answer(client, query, model=MODEL)
        st.write(answer)

    # Button to quit the app    
    if st.button('Quit'):
        st.write('Terminating the app...')
        st.stop()

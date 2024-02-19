"""
This script demonstrates how to use the OpenAI API to call a function that interacts with an external API,
such as the OpenWeatherMap API. Note to use this script, you will need to set up an account with OpenWeatherMap.

Partly based (and modified) on the example here: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
"""

import openai
from openai import OpenAI
import json
import os
import warnings
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Any

weather_api_key = None

def get_weather_data(params:Dict[Any, Any]=None,
                    api_base:str="http://api.weatherstack.com/current") -> Dict[str, str]:
    
    """
    Retrieves weather data from the OpenWeatherMap API.
    """
    import requests
    url = f"{api_base}"
    response = requests.get(url, params=params)
    return response.json()

# Use Weather API to get the current weather for each location
def get_current_weather(location, units="f") -> Dict[str, str]:
    """Get the current weather in a given location"""
    import requests

    api_base:str="http://api.weatherstack.com/current"
    url = f"{api_base}"
    params = {}
    params["access_key"] = weather_api_key
    params["query"] = location
    params["units"] = units
    response = requests.get(url, params=params).json()
    # Extract the relevant data from the response
    if response.get("error"):
        weather_data = json.dumps({
            "error": response["error"]["info"]
        })
        raise Exception(weather_data)
    weather_data = json.dumps({
        "location": response["location"]["name"],
        "temperature": f"{response['current']['temperature']} {units}",
        "weather": response["current"]["weather_descriptions"][0]
    })
    print(weather_data)
    return weather_data
    
def run_conversation(client: object, model: str) -> object:
    # Step 1: send the messages and available functions to the model
    messages = [{"role": "user", "content": "What's the weather like in three cities: San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "units": {"type": "string", "enum": ["m", "f"]},
                    },
                    "required": ["location", ":units"],
                },
            },
        }
    ]
    # Step 2: send the messages and function call info to the model
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        # tool_choice="auto",  # auto is default, but we'll be explicit
        tool_choice={"type": "function", 
                     "function": {"name": "get_current_weather"}}
    )
    # Get the message and tool calls returned by the model
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    print(f"tool_calls: {tool_calls}")
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions_table = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply

        # Step 4: Iterate and send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions_table[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                units=function_args.get("units", "f")
            )
            print(f"function_response: {function_response}")
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        # Step 5: send the function response messages and function call info to the model
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )  # get a new response from the model where it can see the function response
           # and send a new formated message for the user
        return second_response
    else:
        return response
    
if __name__ == "__main__":
    _ = load_dotenv(find_dotenv()) # read local .env file
    warnings.filterwarnings('ignore')
    openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
    openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
    weather_api_key = os.getenv("WEATHER_API_KEY")
    MODEL = os.getenv("MODEL")
    print(f"Using MODEL={MODEL}; base={openai.api_base}")

    client = OpenAI(
        api_key = openai.api_key,
        base_url = openai.api_base
    )
    response = run_conversation(client, MODEL)
    print(response.choices[0].message.content)
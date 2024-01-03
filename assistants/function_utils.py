import os
import math
import random
import warnings

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
from typing import List, Any

import ray
from typing import List, Dict

NUM_OF_CORES = None

warnings.filterwarnings('ignore')
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
weather_api_key = os.getenv("WEATHER_API_KEY")
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base)

def get_num_of_cores():
    """
    Returns the number of cores on the machine."""
    global NUM_OF_CORES
    if NUM_OF_CORES is None:
        NUM_OF_CORES = os.cpu_count()
    return NUM_OF_CORES 

def sampling_task(num_samples: int, task_id: int, verbose=True) -> int:
    num_inside = 0
    for i in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        # check if the point is inside the circle
        if math.hypot(x, y) <= 1:
            num_inside += 1
    if verbose:
        print(f"Task id: {task_id} | Samples in the circle: {num_inside}")
    return num_inside

@ray.remote
def sample_task_distribute(sample_size, i) -> object:
    """
     A Ray remote function that samples points and returns the number of points """
    return sampling_task(sample_size, i)

def run_disributed(sample_size) -> List[int]:
    """
    Runs sampling_task_distribute in parallel using Ray."""
    # Launch Ray remote tasks in a comprehension list, each returns immediately with a future ObjectRef 
    # Use ray.get to fetch the computed value; this will block until the ObjectRef is resolved or its value is materialized.
    results = ray.get([
            sample_task_distribute.remote(sample_size, i+1) for i in range(get_num_of_cores())
        ])
    pi = calculate_pi(results, sample_size)
    return pi

def calculate_pi(results: List[int], sample_size:int) -> float:
    """
    Calculates pi from the results of the sampling tasks."""
    TOTAL_NUM_SAMPLES = get_num_of_cores() * sample_size
    total_num_inside = sum(results)
    pi = (total_num_inside * 4) / TOTAL_NUM_SAMPLES
    return pi

# Define a function which will be fed a dictionary with a key
# holding a list of randomly generated prime numbers between
# 2 and 100

def add_prime_numbers(p_numbers: Dict[str, List[int]]) -> int:
    return sum(p_numbers["prime_numbers"])

def create_dalle_image(params,
                    model:str="dall-e-3" ,
                    quality:str="standard") -> str:
    """"
    Generates an image from the OpenAI DALL-E model."""
    image_description = params["query"]
    response = client.images.generate(model=model,
                                      prompt=image_description,
                                      size="1024x1024",
                                      quality=quality,
                                      n=1)
    
    return response.data[0].url

def get_weather_data(params:Dict[Any, Any]=None,
                    api_base:str="http://api.weatherstack.com/current") -> Dict[str, str]:
    
    """
    Retrieves weather data from the OpenWeatherMap API.
    """
    import requests
    url = f"{api_base}"
    response = requests.get(url, params=params)
    return response.json()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    _ = load_dotenv(find_dotenv()) # read local .env file

    openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
    openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
    weather_api_key = os.getenv("WEATHER_API_KEY")
    MODEL = os.getenv("MODEL")
    print(f"Using MODEL={MODEL}; base={openai.api_base}")

    client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base)

    # Run the sampling task locally
    sample_size = 10_000_000
    pi = run_disributed(sample_size)
    print(f"Estimated value of π is: {pi:5f}")

    #Generate an image from the DALL-E model
    image_description = """A young attractive couple of mixed african and east-indian 
    racial heritage, both of them wearing a matching light fabric summer scarve, sitting 
    at an outside cafe table having a cup of coffee together with the San Francisco Golden 
    Gate Bridge in the background while the sun is setting in the west. The sunset lights 
    up the sky with a beautiful orange glow, partly reflecting on the body of water under the bridge.
    To the right of the couple on the wall is a hanging sign with the name of the Caffe Golden Gate."""

    params = {"query": image_description}
    image_url = create_dalle_image(params, quality="hd", model="dall-e-3")
    print(f"Image URL: {image_url}")

    # Get weather data from the OpenWeatherMap API
    params = {"access_key": weather_api_key,
              "query": "San Francisco",
              "units": "f"}
    print(params)
    weather_data = get_weather_data(params)
    print(f"Weather data for City: {params['query']}")
    print(f"Temperature            : {weather_data['current']['temperature']}")
    print(f"Weather description    : {weather_data['current']['weather_descriptions']}")
    

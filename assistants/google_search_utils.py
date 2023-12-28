import requests
import os
import warnings
import json
from urllib.parse import quote_plus
from pprint import pprint
from typing import List, Dict, Any

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

import requests

def google_search(params:Dict[Any,Any]) -> List[Dict[str, str]]:
    query = quote_plus(params["query"])
    api_key = params["api_key"]
    num_in_page = params["num_results"]
    
    url = f"https://api.apilayer.com/google_search?q={query}"

    payload = {}
    headers= {"apikey": api_key}

    response = requests.request("GET", url, headers=headers, data = payload)

    result = json.loads(response.text)["organic"]
    results = []
    for r in result:
        results.append({
            "title": r["title"],
            "description": r["description"],
            "link": r["link"]
        })
    if len(results) < num_in_page:
        raise ValueError(f"Number of results is less than {num_in_page}")
    return results[0:num_in_page]

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    _ = load_dotenv(find_dotenv()) # read local .env file

    openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
    openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
    google_api_key = os.getenv("GOOGLE_API_KEY")
    MODEL = os.getenv("MODEL")
    print(f"Using MODEL={MODEL}; base={openai.api_base}")

    client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base)
    params = {
        "query": "Top Five best Italian restaurants in San Francisco",
        "api_key": google_api_key,
        "num_results": 5
    }
    results = google_search(params)
    pprint(results)
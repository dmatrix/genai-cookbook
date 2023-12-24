import requests
import json
from urllib.parse import quote_plus
from pprint import pprint

import requests

def google_search(query, key, num_in_page=1):
    query = quote_plus(query)
    
    url = f"https://api.apilayer.com/google_search?q={query}"

    payload = {}
    headers= {"apikey": key}

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
    results = google_search("Top Five best Italian restaurants in San Francisco",
                            "YOUR_GOOGLE_API_KEY", 5)
    pprint(results)
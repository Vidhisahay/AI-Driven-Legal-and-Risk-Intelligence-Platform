API_KEY = "c1011f47873df53d6536d40cb543c1f2009c3d3b"

import requests
import pandas as pd

BASE_URL = "https://www.courtlistener.com/api/rest/v3/opinions/"

def fetch_cases(page_size=200):

    params = {"page_size": page_size}

    headers = {
        "Authorization": f"Token {API_KEY}"
    }

    response = requests.get(BASE_URL, headers=headers, params=params)

    data = response.json()

    print("First result structure:")
    print(data["results"][0])

    return data["results"]


def run_pipeline():

    print("Fetching legal cases from CourtListener API...")

    cases = fetch_cases()

    df = pd.DataFrame(cases)

    df.to_csv("data/raw/legal_cases.csv", index=False)

    print("Saved dataset:", len(df))


if __name__ == "__main__":
    run_pipeline()
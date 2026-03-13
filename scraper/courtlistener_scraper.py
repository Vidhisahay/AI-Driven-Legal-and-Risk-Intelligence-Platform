import requests
import pandas as pd
import time

API_KEY = "c1011f47873df53d6536d40cb543c1f2009c3d3b"

BASE_URL = "https://www.courtlistener.com/api/rest/v4/opinions/"
OUTPUT_FILE = "data/raw/legal_cases.csv"

headers = {
    "Authorization": f"Token {API_KEY}",
    "User-Agent": "Vidhi-Legal-AI-Research-Project"
}


def fetch_cases(pages=50, page_size=200):

    all_cases = []

    for page in range(1, pages + 1):

        print(f"Fetching page {page}")

        params = {
            "page_size": page_size,
            "page": page
        }

        try:

            response = requests.get(BASE_URL, headers=headers, params=params)

            if response.status_code != 200:
                print("API error:", response.status_code)
                time.sleep(5)
                continue

            data = response.json()

            if "results" not in data:
                print("Unexpected API response:", data)
                break

            for result in data["results"]:

                case = {
                    "case_name": result.get("absolute_url"),
                    "date": result.get("date_created"),
                    "court": result.get("cluster"),
                    "text": result.get("plain_text")
                }

                if case["text"] and len(case["text"]) > 500:
                    all_cases.append(case)

            print("Collected cases so far:", len(all_cases))

            time.sleep(2)

        except Exception as e:

            print("Error occurred:", e)
            time.sleep(5)

    return all_cases


def run_pipeline():

    print("Starting legal data ingestion pipeline")

    cases = fetch_cases()

    df = pd.DataFrame(cases)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset saved")
    print("Total legal documents:", len(df))


if __name__ == "__main__":
    run_pipeline()
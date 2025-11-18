import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
dune_api_key = os.getenv("DEFI_JOSH_DUNE_QUERY_API_KEY")

query_id = "6221750"    # Data source: https://dune.com/queries/6221750

url = f"https://api.dune.com/api/v1/query/{query_id}/results/csv"

headers = {"X-DUNE-API-KEY": dune_api_key}

response = requests.get(url, headers=headers)

# Save to CSV file
with open("ronin_data.csv", "w") as file:
    file.write(response.text)

print("Data saved to ronin_data.csv")
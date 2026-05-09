import requests

key = "GB61oVXOg79jXESaDeuv"
url = f"https://api.roboflow.com/minoshs-workspace-vqmme?api_key={key}"
r = requests.get(url)
print(r.json())

import json
import urllib.request
import urllib.error

# The URL where your API is running
url = "http://127.0.0.1:5004/api/v1/valuate"

# The Authorization Bearer token from your .env
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dev-token-change-in-production"
}

# The payload (combining vehicle details + scores from other components)
payload = {
    "maf_year": 2015,
    "reg_year": 2015,
    "mileage_km": 85000,
    "previous_owners": 2,
    "is_reconditioned": 0,
    "power_shutters": 1,
    "power_mirrors": 0,
    "listed_price_million": 3.89,
    
    # Mock scores from Component 3 (Engine)
    "fault_class": "knocking",
    "confidence": 0.87,
    
    # Mock score from Component 2 (Body)
    "body_score": 82,
    
    # Mock score from Component 1 (VIN)
    "vin_status": "original"
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers=headers, method='POST')

print("Sending request to Component 4 API...")
try:
    with urllib.request.urlopen(req) as response:
        print(f"Status Code: {response.status}")
        
        # Read the response
        response_body = response.read().decode('utf-8')
        response_json = json.loads(response_body)
        
        # Print the beautifully formatted JSON response
        print("\nResponse:")
        print(json.dumps(response_json, indent=4))

except urllib.error.URLError as e:
    print(f"Failed to connect to the API. Error: {e.reason}")
    if hasattr(e, 'read'):
        print(e.read().decode('utf-8'))

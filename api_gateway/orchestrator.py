import json
import urllib.request
import urllib.error
import time

# --- Configuration ---
# Update these if you change the ports of your components
API_C1_VIN = "http://127.0.0.1:8000/api/v1/predict"
API_C2_BODY = "http://127.0.0.1:8080/api/v1/analyze"
API_C3_ENGINE = "http://127.0.0.1:5003/api/v1/analyze"
API_C4_VALUATION = "http://127.0.0.1:5004/api/v1/valuate"

# Auth tokens if required
COMMON_HEADERS = {"Authorization": "Bearer dev-token-change-in-production"}
C4_HEADERS = {"Authorization": "Bearer dev-token-change-in-production", "Content-Type": "application/json"}

def print_step(title):
    print("\n" + "="*50)
    print(f">> {title}")
    print("="*50)

def ping_c1_vin(image_path=None):
    """Component 1: VIN Authentication"""
    print_step("Step 1: Pinging Component 1 (VIN Authentication)")
    print(f"Sending VIN image to {API_C1_VIN}...")
    
    # In a real app, you would send the image file via multipart/form-data here.
    # For this demo, we simulate a connection attempt and gracefully fallback.
    try:
        # Attempt to ping (this will likely 405 or 400 without the right payload, triggering fallback)
        req = urllib.request.Request(API_C1_VIN, headers=COMMON_HEADERS, method='POST')
        urllib.request.urlopen(req, timeout=2)
        # If we somehow succeed, we'd parse the JSON
        # return response_json.get("prediction", "original")
    except Exception as e:
        print(f"[WARN] C1 unreachable or missing payload ({e}). Using FALLBACK data.")
    
    # Fallback
    fallback_status = "original"
    print(f"[OK] Result -> VIN Status: {fallback_status}")
    return fallback_status

def ping_c2_body(images=None):
    """Component 2: Body Condition Analysis"""
    print_step("Step 2: Pinging Component 2 (Body Condition)")
    print(f"Sending 5 vehicle views to {API_C2_BODY}...")
    
    try:
        req = urllib.request.Request(API_C2_BODY, headers=COMMON_HEADERS, method='POST')
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        print(f"[WARN] C2 unreachable or missing payload ({e}). Using FALLBACK data.")
    
    # Fallback
    fallback_score = 85
    print(f"[OK] Result -> Body Condition Score: {fallback_score}")
    return fallback_score

def ping_c3_engine(audio_path=None):
    """Component 3: Engine Audio Analysis"""
    print_step("Step 3: Pinging Component 3 (Engine Audio)")
    print(f"Sending engine audio to {API_C3_ENGINE}...")
    
    try:
        req = urllib.request.Request(API_C3_ENGINE, headers=C3_HEADERS, method='POST')
        urllib.request.urlopen(req, timeout=2)
    except Exception as e:
        print(f"[WARN] C3 unreachable or missing payload ({e}). Using FALLBACK data.")
    
    # Fallback
    fallback_fault = "knocking" # Simulating a fault to see deduction
    fallback_conf = 0.92
    print(f"[OK] Result -> Engine Fault: {fallback_fault} (Confidence: {fallback_conf})")
    return fallback_fault, fallback_conf

def ping_c4_valuation(vehicle_data, vin_status, body_score, fault_class, confidence):
    """Component 4: Final Market Valuation"""
    print_step("Step 4: Pinging Component 4 (Market Valuation)")
    print(f"Sending combined aggregate data to {API_C4_VALUATION}...")
    
    # Merge everything into the payload Component 4 expects
    payload = vehicle_data.copy()
    payload.update({
        "vin_status": vin_status,
        "body_score": body_score,
        "fault_class": fault_class,
        "confidence": confidence
    })
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(API_C4_VALUATION, data=data, headers=C4_HEADERS, method='POST')
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                response_body = response.read().decode('utf-8')
                result = json.loads(response_body)
                print("[OK] Component 4 Success! Market Value Calculated.\n")
                return result
            else:
                print(f"[ERROR] Component 4 returned status {response.status}")
                return None
    except Exception as e:
        print(f"[CRITICAL ERROR] Component 4 failed! Make sure it is running on port 5004. Error: {e}")
        if hasattr(e, 'read'):
            print(e.read().decode('utf-8'))
        return None

def run_full_inspection():
    print("--- Starting Automated Vehicle Inspection & Valuation System ---")
    time.sleep(1)
    
    # Base vehicle information submitted by user
    vehicle_data = {
        "maf_year": 2016,
        "reg_year": 2016,
        "mileage_km": 60000,
        "previous_owners": 1,
        "is_reconditioned": 1,
        "power_shutters": 1,
        "power_mirrors": 1,
        "listed_price_million": 4.5
    }
    
    # Step 1: VIN Check
    vin_status = ping_c1_vin()
    time.sleep(1)
    
    # Step 2: Body Check
    body_score = ping_c2_body()
    time.sleep(1)
    
    # Step 3: Engine Check
    fault_class, conf = ping_c3_engine()
    time.sleep(1)
    
    # Step 4: Final Valuation
    result = ping_c4_valuation(vehicle_data, vin_status, body_score, fault_class, conf)
    
    # Display Final Output cleanly
    if result:
        print_step("FINAL VALUATION REPORT")
        print(f"Listed Price:        LKR {result['listed_price_lkr']:,}")
        print(f"Base Market Value:   LKR {result['base_market_value_lkr']:,}")
        print(f"Body Damage Deduct: -LKR {result['body_deduction_lkr']:,}")
        print(f"Engine Fix Deduct:  -LKR {result['engine_deduction_lkr']:,}")
        print("-" * 40)
        print(f"Fair Market Value:   LKR {result['fair_value_lkr']:,}")
        print("-" * 40)
        print(f"\nVerdict: {result['verdict']}")
        print(f"Explanation: {result['explanation']}")

if __name__ == "__main__":
    run_full_inspection()

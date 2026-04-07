import requests
import json

base_url = "http://127.0.0.1:7860"

try:
    print("Testing /reset...")
    resp = requests.post(f"{base_url}/reset", json={})
    print(resp.json())
    
    print("\nTesting /step...")
    action = {
        "category": "account",
        "priority": "low",
        "action": "ignore"
    }
    resp = requests.post(f"{base_url}/step", json=action)
    print(resp.json())

except Exception as e:
    print(f"Error: {e}")

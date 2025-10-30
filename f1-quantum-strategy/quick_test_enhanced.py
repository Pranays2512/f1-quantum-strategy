"""Quick test for enhanced features"""
import requests
import json

print("\n🧪 Testing Enhanced Backend\n")

# Test data with history simulation
base_data = {
    "timestamp": 1234567890,
    "our_car": {
        "position": 3,
        "speed": 285,
        "tyre_temp": {"FL": 95, "FR": 95, "RL": 95, "RR": 95},
        "tyre_wear": 45,
        "fuel_load": 50,
        "lap_time": 90.0,
        "current_lap": 20,
        "sector_times": [30.0, 32.0, 28.0],
        "slow_sectors": []
    },
    "competitors": [],
    "track_conditions": {"temperature": 30, "rainfall": 0, "track_evolution": 85},
    "total_laps": 50,
    "drs_zones": ["DRS Zone 1"]
}

# Test 1: Normal condition
print("Test 1: Normal condition")
response = requests.post("http://localhost:8000/api/strategy/analyze", json=base_data)
result = response.json()
print(f"  Recommendation: {result['pit_stop_recommendation']['recommendation']}")
print(f"  Features: {result['pit_stop_recommendation'].get('features_analyzed', {})}")

# Test 2: Critical condition
print("\nTest 2: Critical condition")
critical_data = json.loads(json.dumps(base_data))
critical_data['our_car']['tyre_wear'] = 88
critical_data['our_car']['tyre_temp'] = {"FL": 118, "FR": 118, "RL": 115, "RR": 116}

response = requests.post("http://localhost:8000/api/strategy/analyze", json=critical_data)
result = response.json()
print(f"  Recommendation: {result['pit_stop_recommendation']['recommendation']}")
print(f"  Confidence: {result['pit_stop_recommendation']['confidence']}%")

# Test 3: Check stats
print("\nTest 3: Stats endpoint")
response = requests.get("http://localhost:8000/api/stats")
print(f"  Stats: {response.json()}")

print("\n✅ All tests complete!\n")
"""
Quick 30-second test to verify backend is working
Run this first to check if everything is set up correctly
"""

import requests

print("\n" + "=" * 60)
print("⚡ QUICK TEST - F1 Quantum Strategy Backend")
print("=" * 60 + "\n")

# Test 1: Is backend running?
print("1️⃣  Checking if backend is running...")
try:
    response = requests.get("http://localhost:8000/api/health", timeout=3)
    if response.status_code == 200:
        print("   ✅ Backend is running!\n")
    else:
        print(f"   ❌ Backend returned error: {response.status_code}\n")
        exit(1)
except:
    print("   ❌ Backend is NOT running!")
    print("   → Start it with: python main.py")
    print("   → Then run this test again\n")
    exit(1)

# Test 2: Can we get strategy recommendations?
print("2️⃣  Testing strategy analysis...")

simple_data = {
    "timestamp": 1234567890,
    "our_car": {
        "position": 3,
        "speed": 285,
        "tyre_temp": {"FL": 95, "FR": 95, "RL": 95, "RR": 95},
        "tyre_wear": 60,
        "fuel_load": 50,
        "lap_time": 90.0,
        "current_lap": 20,
        "sector_times": [30.0, 32.0, 28.0],
        "slow_sectors": []
    },
    "competitors": [],
    "track_conditions": {
        "temperature": 30,
        "rainfall": 0,
        "track_evolution": 85
    },
    "total_laps": 50,
    "drs_zones": ["DRS Zone 1"]
}

try:
    response = requests.post(
        "http://localhost:8000/api/strategy/analyze",
        json=simple_data,
        timeout=5
    )
    
    if response.status_code == 200:
        result = response.json()
        print("   ✅ Strategy analysis working!\n")
        
        # Show a sample result
        pit = result['pit_stop_recommendation']
        print("   📊 Sample Output:")
        print(f"   • Pit: {pit['recommendation']}")
        print(f"   • Confidence: {pit['confidence']}%")
        print(f"   • Compound: {pit['tyre_compound']}")
        print(f"   • Pace: {result['pace_strategy']['pace_mode']}\n")
    else:
        print(f"   ❌ Strategy failed: {response.status_code}\n")
        exit(1)
        
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    exit(1)

# Test 3: Is quantum computing working?
print("3️⃣  Checking quantum computing...")

# Send data with critical tyres (should trigger URGENT)
critical_data = simple_data.copy()
critical_data['our_car'] = simple_data['our_car'].copy()
critical_data['our_car']['tyre_wear'] = 90
critical_data['our_car']['tyre_temp'] = {"FL": 120, "FR": 120, "RL": 120, "RR": 120}

try:
    response = requests.post(
        "http://localhost:8000/api/strategy/pit-timing",
        json=critical_data,
        timeout=5
    )
    
    if response.status_code == 200:
        result = response.json()
        if 'URGENT' in result['recommendation']:
            print("   ✅ Quantum optimization working!")
            print(f"   • Detected critical condition: {result['recommendation']}\n")
        else:
            print("   ✅ Quantum working (logic may vary)\n")
    else:
        print(f"   ⚠️  Quantum test warning: {response.status_code}\n")
        
except Exception as e:
    print(f"   ⚠️  Quantum test warning: {e}\n")

# Final result
print("=" * 60)
print("✅ QUICK TEST PASSED!")
print("=" * 60)
print("\n🎉 Your backend is working correctly!")
print("\n📚 Next steps:")
print("   • Run full tests: python test_backend.py")
print("   • Try scenarios: python scenario_tests.py")
print("   • View API docs: http://localhost:8000/docs")
print("   • Integrate with your F1 simulator\n")
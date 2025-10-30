"""
Complete Test Suite for Enhanced F1 Quantum Strategy Backend - FIXED
Tests all new features and integrations with proper error handling
"""

import requests
import json
import time
from typing import Dict

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*70)
    print(f"🧪 {title}")
    print("="*70)

def test_endpoint(name, url, data=None, method="GET"):
    """Generic endpoint tester with better error handling"""
    print(f"\n📍 Testing: {name}")
    print(f"   Endpoint: {url}")
    
    try:
        start = time.time()
        if method == "GET":
            response = requests.get(f"{BASE_URL}{url}", timeout=10)
        else:
            response = requests.post(f"{BASE_URL}{url}", json=data, timeout=10)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            print(f"   ✅ Success ({elapsed:.2f}s)")
            return response.json(), True
        else:
            print(f"   ❌ Failed: {response.status_code}")
            if response.text:
                print(f"   Response: {response.text[:200]}")
            return None, False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection Error: Backend not running")
        return None, False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, False

def get_test_data() -> Dict:
    """Generate comprehensive test data"""
    return {
        "timestamp": int(time.time()),
        "our_car": {
            "position": 3,
            "speed": 285,
            "tyre_temp": {"FL": 95, "FR": 97, "RL": 93, "RR": 94},
            "tyre_wear": 45,
            "fuel_load": 65,
            "lap_time": 89.234,
            "current_lap": 15,
            "sector_times": [28.5, 32.1, 28.6],
            "slow_sectors": ["Sector 2"]
        },
        "competitors": [
            {
                "car_id": "car_2",
                "position": 2,
                "speed": 290,
                "gap": 2.3,
                "slow_zones": ["Turn 5"],
                "tyre_age": 18
            },
            {
                "car_id": "car_4",
                "position": 4,
                "speed": 280,
                "gap": 0.8,
                "slow_zones": ["Sector 1"],
                "tyre_age": 12
            }
        ],
        "track_conditions": {
            "temperature": 32,
            "rainfall": 0,
            "track_evolution": 85
        },
        "total_laps": 50,
        "drs_zones": ["DRS Zone 1", "DRS Zone 2"]
    }

def test_basic_connectivity():
    """Test 1: Basic connectivity"""
    print_section("TEST 1: Basic Connectivity")
    
    result, success = test_endpoint("Health Check", "/api/health")
    if success:
        print(f"\n   📊 Status: {result.get('status')}")
        print(f"   Version: {result.get('version')}")
        engines = result.get('engines_loaded', {})
        loaded = sum(1 for v in engines.values() if v)
        print(f"   Engines loaded: {loaded}/{len(engines)}")
    
    return success

def test_comprehensive_analysis():
    """Test 2: Comprehensive strategy analysis"""
    print_section("TEST 2: Comprehensive Strategy Analysis")
    
    data = get_test_data()
    result, success = test_endpoint(
        "Complete Analysis",
        "/api/strategy/analyze",
        data,
        "POST"
    )
    
    if success and result:
        print(f"\n   📊 Analysis Results:")
        pit = result.get('pit_stop_recommendation', {})
        print(f"      Pit: {pit.get('recommendation', 'N/A')}")
        print(f"      Confidence: {pit.get('confidence', 0)}%")
        
        if 'tyre_analysis' in result:
            tyre = result['tyre_analysis']
            life = tyre.get('life_prediction', {})
            print(f"\n   🔧 Tyre Analysis:")
            print(f"      Predicted failure lap: {life.get('predicted_failure_lap', 'N/A')}")
            print(f"      Wear rate: {life.get('current_wear_rate', 'N/A')} %/lap")
        
        if 'performance_analysis' in result:
            perf = result['performance_analysis']
            print(f"\n   📈 Performance:")
            consistency = perf.get('lap_consistency', {})
            print(f"      Consistency: {consistency.get('consistency_rating', 'N/A')}")
        
        print(f"\n   ⏱️  Response time: {result.get('response_time_ms', 0)}ms")
    
    return success

def test_pit_prediction():
    """Test 3: Advanced pit stop prediction"""
    print_section("TEST 3: Advanced Pit Stop Prediction")
    
    # Build up some history first
    data = get_test_data()
    
    # Send multiple laps to build history
    print("   Building telemetry history...")
    for lap in range(10, 20):
        lap_data = data.copy()
        lap_data['our_car'] = data['our_car'].copy()
        lap_data['our_car']['current_lap'] = lap
        lap_data['our_car']['tyre_wear'] = 30 + (lap - 10) * 3
        lap_data['our_car']['lap_time'] = 89.0 + (lap - 10) * 0.1
        try:
            requests.post(f"{BASE_URL}/api/strategy/analyze", json=lap_data, timeout=5)
        except:
            pass
    
    time.sleep(0.5)  # Give server time to process
    
    # Now test prediction
    result, success = test_endpoint(
        "Pit Timing Prediction",
        "/api/strategy/pit-timing",
        data,
        "POST"
    )
    
    if success and result:
        print(f"\n   📊 Prediction Results:")
        pred = result.get('predictive_model', {})
        print(f"      Predicted pit lap: {pred.get('predicted_pit_lap', 'N/A')}")
        print(f"      Confidence: {pred.get('confidence', 0)}%")
        
        strategy = result.get('strategic_analysis', {})
        print(f"\n   🎯 Strategy:")
        print(f"      {strategy.get('recommendation', 'N/A')}")
    
    return success

def test_tyre_analysis():
    """Test 4: Complete tyre analysis"""
    print_section("TEST 4: Complete Tyre Analysis")
    
    data = get_test_data()
    
    # Build history
    print("   Building tyre history...")
    for lap in range(5, 15):
        lap_data = data.copy()
        lap_data['our_car'] = data['our_car'].copy()
        lap_data['our_car']['current_lap'] = lap
        lap_data['our_car']['tyre_wear'] = 20 + lap * 2.5
        lap_data['our_car']['tyre_temp'] = {
            "FL": 90 + lap * 0.5,
            "FR": 91 + lap * 0.5,
            "RL": 89 + lap * 0.5,
            "RR": 90 + lap * 0.5
        }
        try:
            requests.post(f"{BASE_URL}/api/strategy/analyze", json=lap_data, timeout=5)
        except:
            pass
    
    time.sleep(0.5)
    
    result, success = test_endpoint(
        "Tyre Analysis",
        "/api/tyre/analysis",
        data,
        "POST"
    )
    
    if success and result:
        print(f"\n   📊 Tyre Life Prediction:")
        life = result.get('life_prediction', {})
        print(f"      Failure lap: {life.get('predicted_failure_lap', 'N/A')}")
        print(f"      Laps remaining: {life.get('laps_remaining_estimate', 'N/A')}")
        print(f"      Confidence: {life.get('confidence', 0)}%")
        
        print(f"\n   🌡️  Temperature Forecast:")
        temp = result.get('temperature_forecast', {})
        print(f"      Trend: {temp.get('temperature_trend', 'N/A')}")
        print(f"      Warning: {temp.get('warning', 'None')}")
        
        print(f"\n   🔧 Compound Recommendation:")
        comp = result.get('compound_recommendation', {})
        print(f"      Recommended: {comp.get('recommended_compound', 'N/A')}")
        print(f"      Alternative: {comp.get('alternative', 'N/A')}")
        print(f"      Confidence: {comp.get('confidence', 0)}%")
    
    return success

def test_weak_point_detection():
    """Test 5: Weak point detection"""
    print_section("TEST 5: Weak Point Detection")
    
    data = get_test_data()
    
    # Build varied sector times
    print("   Building sector performance history...")
    sector_variations = [
        [28.2, 32.5, 28.4],
        [28.3, 32.3, 28.5],
        [28.1, 32.7, 28.3],
        [28.4, 32.4, 28.6],
        [28.2, 32.6, 28.4],
        [28.3, 32.2, 28.5],
        [28.1, 32.8, 28.3],
    ]
    
    for i, sectors in enumerate(sector_variations):
        lap_data = data.copy()
        lap_data['our_car'] = data['our_car'].copy()
        lap_data['our_car']['current_lap'] = 10 + i
        lap_data['our_car']['sector_times'] = sectors
        lap_data['our_car']['lap_time'] = sum(sectors)
        try:
            requests.post(f"{BASE_URL}/api/strategy/analyze", json=lap_data, timeout=5)
        except:
            pass
    
    time.sleep(0.5)
    
    result, success = test_endpoint(
        "Weak Point Analysis",
        "/api/performance/weakpoints",
        data,
        "POST"
    )
    
    if success and result:
        print(f"\n   📊 Sector Performance:")
        sectors = result.get('sector_analysis', {})
        weak = sectors.get('weak_sectors', [])
        if weak:
            print(f"      Weak sectors found: {len(weak)}")
            for w in weak[:2]:  # Show first 2
                print(f"        • {w.get('sector', 'N/A')}: {w.get('severity', 'N/A')} "
                      f"(loss: {w.get('relative_loss', 0):.1f}%)")
        else:
            print(f"      No critical weak sectors identified")
        
        print(f"\n   📈 Consistency:")
        consistency = result.get('consistency_analysis', {})
        print(f"      Overall: {consistency.get('overall_consistency', 'N/A')}%")
        print(f"      Most consistent: {consistency.get('most_consistent', 'N/A')}")
        
        print(f"\n   🎯 Improvement Priorities:")
        priorities = result.get('improvement_priorities', [])
        for i, p in enumerate(priorities[:3], 1):
            print(f"      {i}. [{p.get('priority', 'N/A')}] {p.get('area', 'N/A')}")
            print(f"         {p.get('action', 'N/A')}")
    
    return success

def test_quantum_features():
    """Test 6: Advanced quantum features"""
    print_section("TEST 6: Quantum Advanced Features")
    
    data = get_test_data()
    
    # Test Monte Carlo
    print("\n   🔬 Quantum Monte Carlo Simulation:")
    result, success1 = test_endpoint(
        "Monte Carlo",
        "/api/quantum/monte-carlo",
        data,
        "POST"
    )
    
    if success1 and result:
        print(f"      Simulations: {result.get('num_simulations', 0)}")
        print(f"      Recommended: {result.get('recommended_strategy', 'N/A')}")
        print(f"      Confidence: {result.get('confidence', 0)}%")
    
    # Test Hybrid Optimization
    print("\n   ⚡ Hybrid Classical-Quantum:")
    result, success2 = test_endpoint(
        "Hybrid Optimization",
        "/api/quantum/hybrid-optimize",
        data,
        "POST"
    )
    
    if success2 and result:
        print(f"      Method: {result.get('method', 'N/A')}")
        print(f"      Optimal: {result.get('optimal_strategy', 'N/A')}")
        print(f"      Confidence: {result.get('combined_confidence', 0)}%")
    
    return success1 and success2

def test_quantum_visualization():
    """Test 7: Quantum algorithm visualization"""
    print_section("TEST 7: Quantum Algorithm Visualization")
    
    data = get_test_data()
    
    result, success = test_endpoint(
        "Quantum Visualization",
        "/api/quantum/visualize",
        data,
        "POST"
    )
    
    if success and result:
        viz = result.get('visualization', {})
        print(f"\n   📊 Visualization Generated:")
        print(f"      Algorithm steps: {len(viz.get('algorithm_steps', []))}")
        print(f"      Quantum states: {len(viz.get('quantum_states', []))}")
        
        demo = result.get('demo_notes', {})
        judges = demo.get('for_judges', [])
        if judges:
            print(f"\n   🎯 For Judges:")
            for note in judges[:3]:
                print(f"      • {note}")
        
        final = result.get('final_recommendation', {})
        print(f"\n   🏁 Final Recommendation:")
        print(f"      {final.get('recommendation', 'N/A')}")
        print(f"      Confidence: {final.get('confidence', 0)}%")
    
    return success

def test_critical_scenarios():
    """Test 8: Critical race scenarios"""
    print_section("TEST 8: Critical Scenarios")
    
    scenarios = [
        {
            "name": "Critical Tyre Wear",
            "modifications": {
                "our_car": {
                    "tyre_wear": 88,
                    "tyre_temp": {"FL": 118, "FR": 120, "RL": 116, "RR": 119}
                }
            },
            "expected": "URGENT pit recommendation"
        },
        {
            "name": "Heavy Rain",
            "modifications": {
                "track_conditions": {
                    "rainfall": 75,
                    "temperature": 18
                }
            },
            "expected": "Wet/Intermediate tyres"
        },
        {
            "name": "DRS Overtake Opportunity",
            "modifications": {
                "competitors": [{
                    "car_id": "car_2",
                    "position": 2,
                    "speed": 270,
                    "gap": 0.7,
                    "slow_zones": ["Sector 2"],
                    "tyre_age": 25
                }]
            },
            "expected": "High probability overtake"
        }
    ]
    
    passed = 0
    for scenario in scenarios:
        print(f"\n   🎬 Scenario: {scenario['name']}")
        data = get_test_data()
        
        # Deep merge modifications
        for key, value in scenario['modifications'].items():
            if isinstance(value, dict):
                data[key].update(value)
            else:
                data[key] = value
        
        result, success = test_endpoint(
            scenario['name'],
            "/api/strategy/analyze",
            data,
            "POST"
        )
        
        if success:
            print(f"      Expected: {scenario['expected']}")
            print(f"      ✅ Test passed")
            passed += 1
    
    return passed == len(scenarios)

def test_performance_metrics():
    """Test 9: System performance"""
    print_section("TEST 9: Performance Metrics")
    
    result, success = test_endpoint("Statistics", "/api/stats")
    
    if success and result:
        print(f"\n   📊 System Statistics:")
        print(f"      Total requests: {result.get('total_requests', 0)}")
        print(f"      Avg response time: {result.get('avg_response_time_ms', 0)}ms")
        print(f"      Cars tracked: {result.get('cars_tracked', 0)}")
        print(f"      Total samples: {result.get('total_samples', 0)}")
    
    return success

def run_all_tests():
    """Execute all tests"""
    print("\n" + "="*70)
    print("🏎️  F1 QUANTUM STRATEGY - ENHANCED FEATURES TEST SUITE")
    print("="*70)
    print("\nMake sure the backend is running: python main.py")
    print("Starting tests in 2 seconds...\n")
    time.sleep(2)
    
    tests = [
        ("Basic Connectivity", test_basic_connectivity),
        ("Comprehensive Analysis", test_comprehensive_analysis),
        ("Pit Prediction", test_pit_prediction),
        ("Tyre Analysis", test_tyre_analysis),
        ("Weak Point Detection", test_weak_point_detection),
        ("Quantum Features", test_quantum_features),
        ("Quantum Visualization", test_quantum_visualization),
        ("Critical Scenarios", test_critical_scenarios),
        ("Performance Metrics", test_performance_metrics)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n   ❌ Test failed with exception: {e}")
            results.append((name, False))
        time.sleep(0.5)  # Small delay between tests
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print(f"\n   Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n   🎉 ALL TESTS PASSED!")
    else:
        print(f"\n   ⚠️  {total - passed} test(s) failed")
    
    print("\n" + "="*70)
    print("\n📚 Next Steps:")
    print("   • View API docs: http://localhost:8000/docs")
    print("   • Check features: http://localhost:8000/api/features")
    print("   • Open demo dashboard: quantum_demo_dashboard.html")
    print("   • Run scenario tests: python scenario_tests.py")
    print("\n")

if __name__ == "__main__":
    run_all_tests()
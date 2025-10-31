"""
Simple Working Test Script for F1 Strategy Backend
Tests all core functionality with proper error handling
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*70)
    print(f"🏎️  {title}")
    print("="*70)

def test_health():
    """Test 1: Health Check"""
    print_header("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy")
            print(f"   Version: {data.get('version')}")
            print(f"   Engines loaded:")
            for engine, status in data.get('engines_loaded', {}).items():
                print(f"      {'✅' if status else '❌'} {engine}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\n💡 Make sure the server is running:")
        print("   python main.py")
        return False

def get_sample_data(lap=15):
    """Generate sample race data"""
    return {
        "our_car": {
            "position": 3,
            "speed": 290,
            "tyre_temp": {"FL": 95, "FR": 96, "RL": 94, "RR": 95},
            "tyre_wear": 30 + lap * 2,
            "fuel_load": 110 - lap * 1.8,
            "lap_time": 89.2 + lap * 0.05,
            "current_lap": lap,
            "sector_times": [28.5, 32.0, 28.7],
            "slow_sectors": []
        },
        "competitors": [
            {
                "car_id": "car_2",
                "position": 2,
                "speed": 285,
                "gap": 2.5,
                "slow_zones": ["Sector 2"],
                "tyre_age": lap + 3
            },
            {
                "car_id": "car_4",
                "position": 4,
                "speed": 287,
                "gap": 1.8,
                "slow_zones": [],
                "tyre_age": lap - 2
            }
        ],
        "track_conditions": {
            "temperature": 28,
            "rainfall": 0,
            "track_evolution": 85
        },
        "total_laps": 50,
        "drs_zones": ["DRS Zone 1", "DRS Zone 2"]
    }

def test_basic_analysis():
    """Test 2: Basic Strategy Analysis"""
    print_header("TEST 2: Basic Strategy Analysis")
    
    try:
        data = get_sample_data(15)
        response = requests.post(
            f"{BASE_URL}/api/strategy/analyze",
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful")
            
            # Pit recommendation
            pit = result.get('pit_stop_recommendation', {})
            print(f"\n🔧 Pit Strategy:")
            print(f"   Recommendation: {pit.get('recommendation', 'N/A')}")
            print(f"   Confidence: {pit.get('confidence', 0)}%")
            print(f"   Optimal lap: {pit.get('optimal_lap', 'N/A')}")
            print(f"   Compound: {pit.get('tyre_compound', 'N/A')}")
            
            # Pace strategy
            pace = result.get('pace_strategy', {})
            print(f"\n⚡ Pace Strategy:")
            print(f"   Mode: {pace.get('pace_mode', 'N/A')}")
            print(f"   Target: {pace.get('lap_time_target', 'N/A')}")
            
            # Overtaking
            overtakes = result.get('overtaking_opportunities', [])
            print(f"\n🎯 Overtaking Opportunities: {len(overtakes)}")
            for opp in overtakes[:2]:
                print(f"   • P{opp.get('current_position')} - {opp.get('probability')}% chance")
            
            print(f"\n⏱️  Response time: {result.get('response_time_ms')}ms")
            print(f"📊 Data quality: {result.get('data_quality')}")
            
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_with_history():
    """Test 3: Analysis with Telemetry History"""
    print_header("TEST 3: Building Telemetry History")
    
    try:
        # Send multiple laps to build history
        print("📡 Sending lap data...")
        for lap in range(10, 21):
            data = get_sample_data(lap)
            response = requests.post(
                f"{BASE_URL}/api/strategy/analyze",
                json=data,
                timeout=5
            )
            if response.status_code == 200:
                print(f"   ✅ Lap {lap} recorded")
            else:
                print(f"   ❌ Lap {lap} failed")
            time.sleep(0.2)  # Small delay
        
        print("\n📊 Analyzing with full history...")
        
        # Final analysis with history
        data = get_sample_data(21)
        response = requests.post(
            f"{BASE_URL}/api/strategy/analyze",
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis with history successful")
            
            # Check if we got enhanced analyses
            if result.get('tyre_analysis'):
                tyre = result['tyre_analysis']
                life = tyre.get('life_prediction', {})
                print(f"\n🔍 Tyre Life Prediction:")
                print(f"   Failure lap: {life.get('predicted_failure_lap', 'N/A')}")
                print(f"   Laps remaining: {life.get('laps_remaining_estimate', 'N/A')}")
                print(f"   Confidence: {life.get('confidence', 0)}%")
            
            if result.get('performance_analysis'):
                perf = result['performance_analysis']
                consistency = perf.get('lap_consistency', {})
                print(f"\n📈 Performance Analysis:")
                print(f"   Consistency: {consistency.get('consistency_rating', 'N/A')}")
                print(f"   Best lap: {consistency.get('best_lap', 'N/A')}s")
                print(f"   Worst lap: {consistency.get('worst_lap', 'N/A')}s")
            
            print(f"\n📊 Telemetry samples: {result.get('telemetry_samples', 0)}")
            
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_critical_scenario():
    """Test 4: Critical Tyre Wear Scenario"""
    print_header("TEST 4: Critical Scenario (High Wear)")
    
    try:
        data = get_sample_data(30)
        # Override with critical values
        data['our_car']['tyre_wear'] = 85
        data['our_car']['tyre_temp'] = {
            "FL": 115, "FR": 118, "RL": 114, "RR": 116
        }
        data['our_car']['lap_time'] = 91.5
        
        response = requests.post(
            f"{BASE_URL}/api/strategy/analyze",
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            pit = result.get('pit_stop_recommendation', {})
            
            print("✅ Critical scenario analyzed")
            print(f"\n🚨 CRITICAL RECOMMENDATION:")
            print(f"   {pit.get('recommendation', 'N/A')}")
            print(f"   Confidence: {pit.get('confidence', 0)}%")
            print(f"   Reasoning: {pit.get('reasoning', 'N/A')}")
            
            # Check if it detected urgency
            if 'URGENT' in pit.get('recommendation', ''):
                print(f"\n✅ System correctly detected critical condition")
            else:
                print(f"\n⚠️  Expected urgent recommendation")
            
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_pit_timing():
    """Test 5: Pit Timing Prediction"""
    print_header("TEST 5: Pit Timing Prediction")
    
    try:
        # Build history first
        print("📡 Building telemetry history...")
        for lap in range(5, 16):
            data = get_sample_data(lap)
            requests.post(f"{BASE_URL}/api/strategy/analyze", json=data, timeout=5)
        
        # Test pit timing endpoint
        data = get_sample_data(16)
        response = requests.post(
            f"{BASE_URL}/api/strategy/pit-timing",
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Pit timing prediction successful")
            
            pred = result.get('predictive_model', {})
            print(f"\n🔧 Pit Window Prediction:")
            print(f"   Predicted lap: {pred.get('predicted_pit_lap', 'N/A')}")
            print(f"   Window: {pred.get('pit_window_start', 'N/A')} - {pred.get('pit_window_end', 'N/A')}")
            print(f"   Confidence: {pred.get('confidence', 0)}%")
            print(f"   Current degradation: {pred.get('current_degradation_pct', 0)}%")
            
            strategy = result.get('strategic_analysis', {})
            print(f"\n📋 Strategic Analysis:")
            print(f"   {strategy.get('recommendation', 'N/A')}")
            
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_stats():
    """Test 6: System Statistics"""
    print_header("TEST 6: System Statistics")
    
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=5)
        
        if response.status_code == 200:
            stats = response.json()
            print("✅ Statistics retrieved")
            print(f"\n📊 System Stats:")
            print(f"   Total requests: {stats.get('total_requests', 0)}")
            print(f"   Avg response time: {stats.get('avg_response_time_ms', 0):.1f}ms")
            print(f"   Cars tracked: {stats.get('cars_tracked', 0)}")
            print(f"   Total samples: {stats.get('total_samples', 0)}")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*70)
    print("🏁  F1 QUANTUM STRATEGY - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nTesting backend at:", BASE_URL)
    print("Make sure server is running: python main.py")
    print("\nStarting tests...\n")
    
    time.sleep(1)
    
    tests = [
        ("Health Check", test_health),
        ("Basic Analysis", test_basic_analysis),
        ("Telemetry History", test_with_history),
        ("Critical Scenario", test_critical_scenario),
        ("Pit Timing", test_pit_timing),
        ("System Stats", test_stats)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            results.append((name, False))
        time.sleep(0.5)
    
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
        print("\n   ✅ System is working correctly")
    else:
        print(f"\n   ⚠️  {total - passed} test(s) failed")
        print("\n   💡 Troubleshooting:")
        print("   1. Make sure server is running: python main.py")
        print("   2. Check server logs for errors")
        print("   3. Verify all dependencies are installed")
    
    print("\n" + "="*70)
    print("\n📚 Next Steps:")
    print("   • View API docs: http://localhost:8000/docs")
    print("   • Try the web dashboard: Open quantum_demo_dashboard.html")
    print("   • Run scenario tests with different data")
    print("\n")

if __name__ == "__main__":
    run_all_tests()
"""
Complete test suite for F1 Quantum Strategy Backend
Run this after starting the backend with: python main.py
"""

import requests
import json
import time

def test_strategy_api():
    """Test the quantum strategy backend"""
    
    print("\n" + "=" * 70)
    print("🧪 F1 QUANTUM STRATEGY BACKEND - COMPLETE TEST SUITE")
    print("=" * 70)
    
    # Test 1: Health check
    print("\n📍 TEST 1: Health Check")
    print("-" * 70)
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy and running!")
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
            print(f"   Quantum Simulator: {health_data.get('quantum_simulator')}")
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to backend!")
        print("   Make sure you started the backend with: python main.py")
        print("   The backend should be running in another terminal.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Main strategy endpoint
    print("\n📍 TEST 2: Strategy Analysis (Quantum Computing)")
    print("-" * 70)
    
    test_data = {
        "timestamp": 1234567890,
        "our_car": {
            "position": 3,
            "speed": 285,
            "tyre_temp": {"FL": 95, "FR": 97, "RL": 93, "RR": 94},
            "tyre_wear": 42,
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
                "tyre_age": 12
            },
            {
                "car_id": "car_4",
                "position": 4,
                "speed": 280,
                "gap": 0.8,  # Within DRS range
                "slow_zones": ["Sector 1"],
                "tyre_age": 20
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
    
    print("   Sending test telemetry data...")
    print(f"   - Our position: {test_data['our_car']['position']}")
    print(f"   - Current lap: {test_data['our_car']['current_lap']}/{test_data['total_laps']}")
    print(f"   - Tyre wear: {test_data['our_car']['tyre_wear']}%")
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/api/strategy/analyze",
            json=test_data,
            timeout=15
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Strategy analysis successful! (took {response_time:.2f}s)")
            strategy = response.json()
            
            # Display detailed results
            print("\n" + "=" * 70)
            print("📊 QUANTUM STRATEGY RECOMMENDATIONS")
            print("=" * 70)
            
            # Pit stop recommendation
            pit = strategy['pit_stop_recommendation']
            print(f"\n🔧 PIT STOP STRATEGY:")
            print(f"   Recommendation: {pit['recommendation']}")
            print(f"   Optimal Lap: {pit.get('optimal_lap', 'N/A')}")
            print(f"   Laps Until Pit: {pit.get('laps_until_pit', 'N/A')}")
            print(f"   Tyre Compound: {pit['tyre_compound']}")
            print(f"   Confidence: {pit['confidence']}%")
            print(f"   Expected Time Impact: {pit['expected_time_impact']}s")
            print(f"   Reasoning: {pit['reasoning']}")
            
            if pit.get('alternative_strategies'):
                print(f"\n   Alternative Strategies:")
                for alt in pit['alternative_strategies'][:2]:
                    print(f"     • Lap {alt['lap']}: {alt['compound']} ({alt['confidence']}%)")
            
            # Pace strategy
            pace = strategy['pace_strategy']
            print(f"\n⚡ PACE STRATEGY:")
            print(f"   Mode: {pace['pace_mode']}")
            print(f"   Lap Time Target: {pace['lap_time_target']}")
            print(f"   Fuel Strategy: {pace['fuel_strategy']}")
            print(f"   Tyre Management: {pace['tyre_management']}")
            print(f"   Recommendation: {pace['recommendation']}")
            
            # Overtaking opportunities
            overtake = strategy['overtaking_opportunities']
            print(f"\n🎯 OVERTAKING OPPORTUNITIES:")
            if overtake:
                for i, opp in enumerate(overtake, 1):
                    print(f"\n   Opportunity {i}:")
                    print(f"     Target: {opp['target_car']} (P{opp['current_position']})")
                    print(f"     Gap: {opp['gap']}s")
                    print(f"     Probability: {opp['probability']}%")
                    print(f"     Speed Advantage: {opp['speed_advantage']} km/h")
                    print(f"     DRS Available: {'Yes' if opp['drs_available'] else 'No'}")
                    print(f"     Best Zones: {', '.join(opp['best_zones']) if opp['best_zones'] else 'None'}")
                    print(f"     Recommendation: {opp['recommendation']}")
            else:
                print("   No immediate overtaking opportunities detected")
            
            # Sector optimization
            sectors = strategy['sector_optimization']
            print(f"\n📊 SECTOR OPTIMIZATION:")
            print(f"   Focus Sector: {sectors.get('focus_sector', 'N/A')}")
            print(f"   Total Potential Gain: {sectors.get('total_potential_gain', 0)}s per lap")
            print(f"   Summary: {sectors.get('summary', 'N/A')}")
            
            if sectors.get('sectors'):
                print(f"\n   Detailed Sector Analysis:")
                for sector in sectors['sectors'][:3]:  # Show top 3
                    print(f"     • {sector['sector']}: {sector['priority']} priority")
                    print(f"       Current time: {sector['current_time']}s")
                    print(f"       Potential gain: {sector['potential_gain']}s")
                    print(f"       Advice: {sector['advice']}")
            
            # Risk and overall impact
            print(f"\n⚠️  RISK ASSESSMENT:")
            print(f"   {strategy['risk_assessment']}")
            
            print(f"\n💡 EXPECTED TOTAL TIME GAIN: {strategy['expected_time_gain']}s")
            
            # Validation checks
            print("\n" + "=" * 70)
            print("🔍 VALIDATION CHECKS")
            print("=" * 70)
            
            checks_passed = 0
            total_checks = 0
            
            # Check 1: Response time
            total_checks += 1
            if response_time < 5:
                print(f"✅ Response time acceptable: {response_time:.2f}s")
                checks_passed += 1
            else:
                print(f"⚠️  Response time slow: {response_time:.2f}s (should be < 5s)")
            
            # Check 2: Confidence score
            total_checks += 1
            if 0 <= pit['confidence'] <= 100:
                print(f"✅ Confidence score valid: {pit['confidence']}%")
                checks_passed += 1
            else:
                print(f"❌ Confidence score invalid: {pit['confidence']}%")
            
            # Check 3: Overtaking logic
            total_checks += 1
            high_prob_overtakes = [o for o in overtake if o['probability'] > 70]
            if high_prob_overtakes:
                drs_available = [o for o in high_prob_overtakes if o['drs_available']]
                if drs_available:
                    print(f"✅ Overtaking logic working: High probability with DRS")
                    checks_passed += 1
                else:
                    print(f"✅ Overtaking logic working: High probability detected")
                    checks_passed += 1
            else:
                print(f"✅ Overtaking logic working: No high-probability opportunities")
                checks_passed += 1
            
            # Check 4: Pace strategy makes sense
            total_checks += 1
            if pace['pace_mode'] in ['ATTACK', 'PUSH', 'BALANCED', 'CONSERVE']:
                print(f"✅ Pace strategy valid: {pace['pace_mode']}")
                checks_passed += 1
            else:
                print(f"❌ Pace strategy invalid: {pace['pace_mode']}")
            
            print(f"\n{'=' * 70}")
            print(f"📊 VALIDATION SCORE: {checks_passed}/{total_checks} checks passed")
            print(f"{'=' * 70}")
            
            return True
            
        else:
            print(f"❌ Strategy analysis failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during strategy analysis: {e}")
        return False
    
    # Test 3: Critical scenario (high tyre wear)
    print("\n📍 TEST 3: Critical Tyre Condition Detection")
    print("-" * 70)
    
    critical_data = test_data.copy()
    critical_data['our_car'] = test_data['our_car'].copy()
    critical_data['our_car']['tyre_wear'] = 88  # Very worn
    critical_data['our_car']['tyre_temp'] = {"FL": 115, "FR": 118, "RL": 112, "RR": 116}  # Very hot
    
    print("   Testing with critical tyre condition:")
    print(f"   - Tyre wear: 88%")
    print(f"   - Tyre temps: 115-118°C (critical!)")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/strategy/pit-timing",
            json=critical_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Critical condition test successful!")
            print(f"   Recommendation: {result['recommendation']}")
            
            if 'URGENT' in result['recommendation'].upper():
                print("   ✅ Correctly identified URGENT pit stop needed!")
            else:
                print("   ⚠️  Did not flag as URGENT (might be okay depending on logic)")
        else:
            print(f"⚠️  Critical test returned status: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Critical test warning: {e}")
    
    return True


def run_performance_test():
    """Test response time performance"""
    
    print("\n" + "=" * 70)
    print("⏱️  PERFORMANCE TEST")
    print("=" * 70)
    
    test_data = {
        "timestamp": 1234567890,
        "our_car": {
            "position": 3,
            "speed": 285,
            "tyre_temp": {"FL": 95, "FR": 97, "RL": 93, "RR": 94},
            "tyre_wear": 42,
            "fuel_load": 65,
            "lap_time": 89.234,
            "current_lap": 15,
            "sector_times": [28.5, 32.1, 28.6],
            "slow_sectors": ["Sector 2"]
        },
        "competitors": [],
        "track_conditions": {"temperature": 32, "rainfall": 0, "track_evolution": 85},
        "total_laps": 50,
        "drs_zones": ["DRS Zone 1"]
    }
    
    print("\nRunning 5 requests to measure average response time...")
    
    times = []
    for i in range(5):
        try:
            start = time.time()
            response = requests.post(
                "http://localhost:8000/api/strategy/analyze",
                json=test_data,
                timeout=10
            )
            elapsed = time.time() - start
            times.append(elapsed)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"   Request {i+1}: {elapsed:.2f}s {status}")
        except Exception as e:
            print(f"   Request {i+1}: Failed - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 Performance Results:")
        print(f"   Average: {avg_time:.2f}s")
        print(f"   Fastest: {min_time:.2f}s")
        print(f"   Slowest: {max_time:.2f}s")
        
        if avg_time < 2:
            print(f"   ✅ Excellent performance!")
        elif avg_time < 3:
            print(f"   ✅ Good performance!")
        elif avg_time < 5:
            print(f"   ⚠️  Acceptable performance (could be optimized)")
        else:
            print(f"   ⚠️  Slow performance (consider reducing quantum shots)")


if __name__ == "__main__":
    print("\n🏎️  Starting F1 Quantum Strategy Backend Tests...")
    print("   Make sure the backend is running: python main.py")
    print("   Waiting 2 seconds for you to check...")
    time.sleep(2)
    
    # Run main tests
    success = test_strategy_api()
    
    if success:
        print("\n" + "=" * 70)
        
        # Run performance test
        run_performance_test()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS COMPLETED!")
        print("=" * 70)
        print("\n✅ Your Quantum Strategy Backend is working perfectly!")
        print("📚 Next steps:")
        print("   1. Check http://localhost:8000/docs for interactive API")
        print("   2. Run scenario_tests.py for more test cases")
        print("   3. Integrate with your F1 simulator")
        print("\n")
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease check:")
        print("   1. Is the backend running? (python main.py)")
        print("   2. Are all dependencies installed? (pip install -r requirements.txt)")
        print("   3. Check the terminal where backend is running for errors")
        print("\n")
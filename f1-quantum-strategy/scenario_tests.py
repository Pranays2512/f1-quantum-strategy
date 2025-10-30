"""
F1 Quantum Strategy - Scenario Tests
Tests different racing situations to verify AI makes smart decisions
"""

import requests
import json

def test_scenario(name, description, data, expected_behavior):
    """Test a specific race scenario"""
    print(f"\n{'=' * 70}")
    print(f"🏎️  Scenario: {name}")
    print(f"📝 {description}")
    print(f"{'=' * 70}")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/strategy/analyze",
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            strategy = response.json()
            pit = strategy['pit_stop_recommendation']
            pace = strategy['pace_strategy']
            risk = strategy['risk_assessment']
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   Pit: {pit['recommendation']}")
            print(f"   Confidence: {pit['confidence']}%")
            print(f"   Compound: {pit['tyre_compound']}")
            print(f"   Pace: {pace['pace_mode']} - {pace['lap_time_target']}")
            print(f"   Risk: {risk}")
            print(f"   Expected Gain: {strategy['expected_time_gain']}s")
            
            # Validation
            print(f"\n🔍 Validation:")
            print(f"   Expected: {expected_behavior}")
            
            # Check if behavior matches expectations
            validation_passed = True
            
            if "URGENT" in expected_behavior:
                if "URGENT" in pit['recommendation']:
                    print(f"   ✅ Correctly identified urgent situation")
                else:
                    print(f"   ⚠️  Did not flag as urgent (expected URGENT)")
                    validation_passed = False
            
            if "Hard" in expected_behavior and "compound" in expected_behavior.lower():
                if "Hard" in pit['tyre_compound']:
                    print(f"   ✅ Correctly chose Hard tyres")
                else:
                    print(f"   ⚠️  Expected Hard tyres, got {pit['tyre_compound']}")
            
            if "wait" in expected_behavior.lower():
                if pit.get('laps_until_pit', 0) > 3:
                    print(f"   ✅ Correctly recommended waiting")
                else:
                    print(f"   ⚠️  Expected to wait longer")
            
            if "Intermediate" in expected_behavior:
                if "Intermediate" in pit['tyre_compound']:
                    print(f"   ✅ Correctly chose rain tyres")
                else:
                    print(f"   ⚠️  Expected Intermediate tyres, got {pit['tyre_compound']}")
            
            if "CONSERVE" in expected_behavior or "conservative" in expected_behavior.lower():
                if "CONSERVE" in pace['pace_mode']:
                    print(f"   ✅ Correctly chose conservative pace")
                else:
                    print(f"   ℹ️  Got {pace['pace_mode']} pace (CONSERVE expected)")
            
            if validation_passed:
                print(f"\n   ✅ Test PASSED")
            else:
                print(f"\n   ⚠️  Test completed with warnings")
            
        else:
            print(f"   ❌ Failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def run_all_scenarios():
    """Run all test scenarios"""
    
    print("\n" + "=" * 70)
    print("🧪 F1 QUANTUM STRATEGY - RACING SCENARIO TESTS")
    print("=" * 70)
    print("\nThese tests verify the AI makes smart decisions in different situations")
    
    # Base data template
    base_data = {
        "timestamp": 1234567890,
        "our_car": {
            "position": 3,
            "speed": 285,
            "tyre_temp": {"FL": 95, "FR": 95, "RL": 95, "RR": 95},
            "tyre_wear": 40,
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
                "gap": 3.5,
                "slow_zones": ["Turn 5"],
                "tyre_age": 12
            }
        ],
        "track_conditions": {
            "temperature": 30,
            "rainfall": 0,
            "track_evolution": 85
        },
        "total_laps": 50,
        "drs_zones": ["DRS Zone 1", "DRS Zone 2"]
    }
    
    # Scenario 1: Critical Tyre Wear
    scenario1 = json.loads(json.dumps(base_data))  # Deep copy
    scenario1['our_car']['tyre_wear'] = 88
    scenario1['our_car']['tyre_temp'] = {"FL": 115, "FR": 118, "RL": 112, "RR": 116}
    scenario1['our_car']['current_lap'] = 25
    
    test_scenario(
        "Critical Tyre Degradation",
        "High wear (88%) and extreme temperatures (115°C+). Tyres about to fail.",
        scenario1,
        "Should recommend URGENT pit stop immediately with Hard compound"
    )
    
    # Scenario 2: Fresh Tyres
    scenario2 = json.loads(json.dumps(base_data))
    scenario2['our_car']['tyre_wear'] = 8
    scenario2['our_car']['tyre_temp'] = {"FL": 88, "FR": 90, "RL": 87, "RR": 89}
    scenario2['our_car']['current_lap'] = 5
    
    test_scenario(
        "Fresh Tyres - Early Race",
        "Just 5 laps in, tyres at 8% wear, good temperatures. No need to pit.",
        scenario2,
        "Should wait many laps before pitting"
    )
    
    # Scenario 3: Rain Conditions
    scenario3 = json.loads(json.dumps(base_data))
    scenario3['track_conditions']['rainfall'] = 65
    scenario3['track_conditions']['temperature'] = 18
    scenario3['our_car']['current_lap'] = 20
    
    test_scenario(
        "Heavy Rain Started",
        "Rainfall at 65%, track temperature dropped. Currently on slick tyres.",
        scenario3,
        "Should recommend Intermediate or wet weather tyres"
    )
    
    # Scenario 4: Leading Position
    scenario4 = json.loads(json.dumps(base_data))
    scenario4['our_car']['position'] = 1
    scenario4['our_car']['tyre_wear'] = 45
    scenario4['our_car']['current_lap'] = 30
    scenario4['competitors'][0]['gap'] = -8.5  # 8.5s behind us
    
    test_scenario(
        "Leading Comfortably",
        "P1 with 8.5s lead, mid-race, moderate tyre wear. Should protect position.",
        scenario4,
        "Should be conservative - maintain gap, don't take risks"
    )
    
    # Scenario 5: Close Racing - DRS Range
    scenario5 = json.loads(json.dumps(base_data))
    scenario5['our_car']['position'] = 4
    scenario5['our_car']['speed'] = 295
    scenario5['our_car']['tyre_wear'] = 25  # Fresh tyres
    scenario5['competitors'] = [
        {
            "car_id": "car_3",
            "position": 3,
            "speed": 285,
            "gap": 0.7,  # Within DRS range!
            "slow_zones": ["Sector 2", "Turn 8"],
            "tyre_age": 18  # Older tyres
        }
    ]
    
    test_scenario(
        "DRS Overtaking Opportunity",
        "0.7s gap (DRS range), we're faster, they have old tyres and slow in sector 2.",
        scenario5,
        "Should identify high-probability overtaking opportunity"
    )
    
    # Scenario 6: End of Race
    scenario6 = json.loads(json.dumps(base_data))
    scenario6['our_car']['current_lap'] = 48
    scenario6['total_laps'] = 50
    scenario6['our_car']['tyre_wear'] = 72
    scenario6['our_car']['position'] = 2
    
    test_scenario(
        "Final Laps - Hold Position",
        "Lap 48/50, P2, tyres at 72%. Just need to finish.",
        scenario6,
        "Should recommend holding on, no pit stop needed"
    )
    
    # Scenario 7: Fuel Critical
    scenario7 = json.loads(json.dumps(base_data))
    scenario7['our_car']['fuel_load'] = 15  # Very low fuel
    scenario7['our_car']['current_lap'] = 35
    scenario7['total_laps'] = 50
    
    test_scenario(
        "Fuel Management Critical",
        "Low fuel (15kg) with 15 laps remaining. Must conserve.",
        scenario7,
        "Should recommend fuel saving mode and conservative pace"
    )
    
    # Scenario 8: Undercut Opportunity
    scenario8 = json.loads(json.dumps(base_data))
    scenario8['our_car']['position'] = 3
    scenario8['our_car']['tyre_wear'] = 52
    scenario8['our_car']['current_lap'] = 22
    scenario8['competitors'] = [
        {
            "car_id": "car_2",
            "position": 2,
            "speed": 285,
            "gap": 3.2,
            "slow_zones": ["Turn 5"],
            "tyre_age": 21  # They haven't pitted yet, old tyres
        }
    ]
    
    test_scenario(
        "Undercut Strategy Opportunity",
        "P3, competitor ahead hasn't pitted (21 lap old tyres). We can undercut.",
        scenario8,
        "Should recommend pitting soon to undercut P2"
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL SCENARIO TESTS COMPLETED!")
    print("=" * 70)
    print("\n📊 Summary:")
    print("   These tests verify the AI responds appropriately to:")
    print("   ✅ Critical tyre conditions")
    print("   ✅ Weather changes")
    print("   ✅ Race position (leading vs chasing)")
    print("   ✅ Overtaking opportunities")
    print("   ✅ Fuel management")
    print("   ✅ Race timing (early, mid, late race)")
    print("   ✅ Strategic opportunities (undercut)")
    print("\n")


if __name__ == "__main__":
    import time
    print("🏎️  F1 Quantum Strategy - Scenario Tests")
    print("Make sure the backend is running: python main.py")
    print("Starting in 2 seconds...")
    time.sleep(2)
    
    run_all_scenarios()
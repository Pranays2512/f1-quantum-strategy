"""
Feature Engineering Module
Extracts meaningful features from telemetry history for better predictions
"""

import numpy as np
from typing import List, Dict

def extract_features_simple(telemetry_samples: List[Dict]) -> Dict:
    """
    Extract features from recent telemetry history
    
    Args:
        telemetry_samples: List of telemetry dictionaries with keys:
            - current_lap, tyre_wear, fuel_load, tyre_temp, lap_time, position
    
    Returns:
        Dictionary of extracted features
    """
    
    if not telemetry_samples:
        return {
            'laps_on_stint': 0,
            'avg_tyre_temp': 95.0,
            'max_tyre_temp': 95.0,
            'temp_trend': 0.0,
            'temp_imbalance': 0.0,
            'wear_rate': 0.0,
            'fuel_usage_rate': 0.0,
            'lap_time_trend': 0.0
        }
    
    # Extract time series data
    laps = [s['current_lap'] for s in telemetry_samples]
    wear_values = [s['tyre_wear'] for s in telemetry_samples]
    fuel_values = [s['fuel_load'] for s in telemetry_samples]
    lap_times = [s['lap_time'] for s in telemetry_samples]
    
    # Extract tyre temps (average of all 4 corners)
    temp_averages = []
    max_temps = []
    temp_imbalances = []
    
    for sample in telemetry_samples:
        temps = sample['tyre_temp']
        temp_list = [temps['FL'], temps['FR'], temps['RL'], temps['RR']]
        temp_averages.append(np.mean(temp_list))
        max_temps.append(max(temp_list))
        temp_imbalances.append(max(temp_list) - min(temp_list))
    
    # Calculate features
    features = {}
    
    # Stint length
    features['laps_on_stint'] = len(telemetry_samples)
    
    # Temperature features
    features['avg_tyre_temp'] = np.mean(temp_averages)
    features['max_tyre_temp'] = max(max_temps)
    features['temp_imbalance'] = np.mean(temp_imbalances)
    
    # Temperature trend (degrees per lap)
    if len(temp_averages) >= 2:
        features['temp_trend'] = (temp_averages[-1] - temp_averages[0]) / len(temp_averages)
    else:
        features['temp_trend'] = 0.0
    
    # Wear rate (% per lap)
    if len(wear_values) >= 2 and laps[-1] != laps[0]:
        features['wear_rate'] = (wear_values[-1] - wear_values[0]) / (laps[-1] - laps[0])
    else:
        features['wear_rate'] = 0.0
    
    # Fuel usage rate (kg per lap)
    if len(fuel_values) >= 2 and laps[-1] != laps[0]:
        features['fuel_usage_rate'] = (fuel_values[0] - fuel_values[-1]) / (laps[-1] - laps[0])
    else:
        features['fuel_usage_rate'] = 0.0
    
    # Lap time trend (improving or degrading)
    if len(lap_times) >= 2:
        features['lap_time_trend'] = lap_times[-1] - lap_times[0]
    else:
        features['lap_time_trend'] = 0.0
    
    return features


def extract_features_advanced(telemetry_samples: List[Dict]) -> Dict:
    """
    Advanced feature extraction with more sophisticated analysis
    
    Includes:
    - Moving averages
    - Rate of change metrics
    - Anomaly detection
    - Predictive indicators
    """
    
    basic_features = extract_features_simple(telemetry_samples)
    
    if len(telemetry_samples) < 5:
        return basic_features
    
    # Advanced features
    advanced = {}
    
    # Extract recent window (last 5 laps)
    recent_window = telemetry_samples[-5:]
    
    # Temperature acceleration (how fast temps are rising)
    temp_values = []
    for sample in telemetry_samples:
        temps = sample['tyre_temp']
        avg_temp = (temps['FL'] + temps['FR'] + temps['RL'] + temps['RR']) / 4
        temp_values.append(avg_temp)
    
    if len(temp_values) >= 3:
        temp_diffs = np.diff(temp_values)
        advanced['temp_acceleration'] = np.mean(temp_diffs[-3:]) - np.mean(temp_diffs[:3]) if len(temp_diffs) >= 3 else 0.0
    else:
        advanced['temp_acceleration'] = 0.0
    
    # Wear acceleration (is wear rate increasing?)
    wear_values = [s['tyre_wear'] for s in telemetry_samples]
    if len(wear_values) >= 3:
        wear_diffs = np.diff(wear_values)
        advanced['wear_acceleration'] = np.mean(wear_diffs[-3:]) - np.mean(wear_diffs[:3]) if len(wear_diffs) >= 3 else 0.0
    else:
        advanced['wear_acceleration'] = 0.0
    
    # Degradation score (0-1, higher = more degraded)
    max_temp = basic_features['max_tyre_temp']
    avg_wear = basic_features.get('wear_rate', 0) * basic_features['laps_on_stint']
    
    temp_score = min(1.0, (max_temp - 90) / 30)  # 90-120°C range
    wear_score = min(1.0, avg_wear / 80)  # 0-80% wear
    
    advanced['degradation_score'] = (temp_score + wear_score) / 2
    
    # Predicted laps remaining (rough estimate)
    if basic_features['wear_rate'] > 0:
        remaining_wear = 100 - (telemetry_samples[-1]['tyre_wear'] if telemetry_samples else 50)
        advanced['predicted_laps_remaining'] = int(remaining_wear / basic_features['wear_rate'])
    else:
        advanced['predicted_laps_remaining'] = 999
    
    # Combine with basic features
    return {**basic_features, **advanced}


def should_pit_urgently(features: Dict) -> bool:
    """
    Decision logic: Should we pit immediately?
    
    Returns True if critical condition detected
    """
    
    critical_conditions = []
    
    # Critical temperature
    if features['max_tyre_temp'] > 115:
        critical_conditions.append("Critical temperature")
    
    # Rapid temperature rise
    if features['temp_trend'] > 3.0:  # More than 3°C per lap
        critical_conditions.append("Rapid temp increase")
    
    # High wear on long stint
    if features['laps_on_stint'] > 20 and features['avg_tyre_temp'] > 105:
        critical_conditions.append("Long stint with high temps")
    
    # Extreme imbalance
    if features['temp_imbalance'] > 15:
        critical_conditions.append("Extreme temp imbalance")
    
    return len(critical_conditions) > 0, critical_conditions


def calculate_confidence_adjustment(features: Dict) -> float:
    """
    Calculate confidence adjustment based on feature analysis
    
    Returns: Adjustment value (-20 to +20)
    """
    
    adjustment = 0.0
    
    # More laps on stint = more confident in prediction
    if features['laps_on_stint'] > 15:
        adjustment += 10
    elif features['laps_on_stint'] < 5:
        adjustment -= 5
    
    # Consistent trends increase confidence
    if abs(features['temp_trend']) > 2:
        adjustment += 5
    
    # High degradation = more confident pit needed
    if features.get('degradation_score', 0) > 0.7:
        adjustment += 10
    
    return max(-20, min(20, adjustment))
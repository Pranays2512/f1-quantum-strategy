"""
Tyre Wear & Temperature Modeling Module
Physical modeling of tyre degradation and temperature dynamics
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit

class TyreModel:
    """Physical model for tyre wear and temperature prediction"""
    
    def __init__(self):
        # Physical constants and empirical coefficients
        self.optimal_temp_range = (90, 105)  # Celsius
        self.critical_temp = 115  # Celsius
        self.base_wear_rate = 2.5  # % per lap at optimal conditions
        self.temp_wear_multiplier = 0.015  # Additional wear per degree above optimal
        self.track_temp_factor = 0.02  # Track temperature influence
        
    def predict_tyre_life(self, telemetry_history: List[Dict],
                         track_temp: float, total_laps: int) -> Dict:
        """
        Predict remaining tyre life and expected wear at future laps
        
        Returns:
            Dictionary with wear predictions and critical lap estimates
        """
        if not telemetry_history:
            return {
                'predicted_failure_lap': None,
                'wear_curve': [],
                'confidence': 0.0
            }
        
        # Extract data
        laps = np.array([s['current_lap'] for s in telemetry_history])
        wear_values = np.array([s['tyre_wear'] for s in telemetry_history])
        
        # Calculate temperatures
        temp_averages = []
        for sample in telemetry_history:
            temps = sample['tyre_temp']
            avg_temp = (temps['FL'] + temps['FR'] + temps['RL'] + temps['RR']) / 4
            temp_averages.append(avg_temp)
        temp_averages = np.array(temp_averages)
        
        # Fit wear model
        wear_rate = self._calculate_dynamic_wear_rate(
            temp_averages, track_temp, laps
        )
        
        # Predict future wear
        current_lap = laps[-1]
        current_wear = wear_values[-1]
        future_laps = np.arange(current_lap + 1, total_laps + 1)
        
        # Build wear curve
        wear_curve = []
        predicted_wear = current_wear
        
        for lap in future_laps:
            # Assume temperature stabilizes near current average
            predicted_temp = np.mean(temp_averages[-3:]) if len(temp_averages) >= 3 else temp_averages[-1]
            lap_wear_rate = self._get_wear_rate_for_conditions(predicted_temp, track_temp)
            predicted_wear += lap_wear_rate
            
            wear_curve.append({
                'lap': int(lap),
                'predicted_wear': round(min(100, predicted_wear), 2),
                'wear_rate': round(lap_wear_rate, 3)
            })
            
            if predicted_wear >= 95:  # Critical wear level
                break
        
        # Find predicted failure lap
        failure_lap = None
        for point in wear_curve:
            if point['predicted_wear'] >= 90:  # Conservative failure threshold
                failure_lap = point['lap']
                break
        
        # Calculate confidence
        confidence = self._calculate_model_confidence(wear_values, laps, wear_rate)
        
        return {
            'predicted_failure_lap': failure_lap,
            'wear_curve': wear_curve[:20],  # Limit to next 20 laps
            'current_wear_rate': round(wear_rate, 3),
            'laps_remaining_estimate': int((90 - current_wear) / wear_rate) if wear_rate > 0 else 999,
            'confidence': round(confidence * 100, 1),
            'model_type': 'Physical degradation model',
            'assumptions': {
                'stable_temperature': round(np.mean(temp_averages[-3:]), 1) if len(temp_averages) >= 3 else None,
                'track_temp': track_temp
            }
        }
    
    def predict_temperature_evolution(self, telemetry_history: List[Dict],
                                     future_laps: int = 10) -> Dict:
        """
        Predict how tyre temperatures will evolve over next laps
        
        Returns:
            Temperature predictions and warnings
        """
        if len(telemetry_history) < 3:
            return {
                'temperature_trend': 'insufficient_data',
                'predictions': []
            }
        
        # Extract temperature history
        laps = []
        avg_temps = []
        max_temps = []
        
        for sample in telemetry_history:
            temps = sample['tyre_temp']
            temp_list = [temps['FL'], temps['FR'], temps['RL'], temps['RR']]
            laps.append(sample['current_lap'])
            avg_temps.append(np.mean(temp_list))
            max_temps.append(max(temp_list))
        
        laps = np.array(laps)
        avg_temps = np.array(avg_temps)
        max_temps = np.array(max_temps)
        
        # Fit temperature evolution curve (exponential approach to equilibrium)
        if len(laps) >= 5:
            try:
                # Exponential model: T(t) = T_eq - (T_eq - T_0) * exp(-k*t)
                def temp_model(lap, t_eq, k, t_0):
                    return t_eq - (t_eq - t_0) * np.exp(-k * (lap - laps[0]))
                
                popt, _ = curve_fit(temp_model, laps, avg_temps, 
                                   p0=[100, 0.1, avg_temps[0]],
                                   maxfev=1000)
                
                # Predict future temperatures
                current_lap = laps[-1]
                future_lap_nums = np.arange(current_lap + 1, current_lap + future_laps + 1)
                predicted_temps = temp_model(future_lap_nums, *popt)
                
                predictions = []
                for i, lap_num in enumerate(future_lap_nums):
                    temp = predicted_temps[i]
                    status = self._get_temp_status(temp)
                    predictions.append({
                        'lap': int(lap_num),
                        'predicted_avg_temp': round(float(temp), 1),
                        'status': status
                    })
                
                # Determine trend
                temp_change = predicted_temps[-1] - avg_temps[-1]
                if temp_change > 5:
                    trend = 'rising'
                elif temp_change < -5:
                    trend = 'cooling'
                else:
                    trend = 'stable'
                
            except:
                # Fallback to linear trend
                temp_slope = (avg_temps[-1] - avg_temps[0]) / (laps[-1] - laps[0])
                predictions = []
                current_temp = avg_temps[-1]
                
                for i in range(1, future_laps + 1):
                    future_temp = current_temp + (temp_slope * i)
                    predictions.append({
                        'lap': int(laps[-1] + i),
                        'predicted_avg_temp': round(float(future_temp), 1),
                        'status': self._get_temp_status(future_temp)
                    })
                
                if temp_slope > 1:
                    trend = 'rising'
                elif temp_slope < -1:
                    trend = 'cooling'
                else:
                    trend = 'stable'
        else:
            # Insufficient data - assume stable
            current_temp = avg_temps[-1]
            predictions = [{
                'lap': int(laps[-1] + i),
                'predicted_avg_temp': round(float(current_temp), 1),
                'status': self._get_temp_status(current_temp)
            } for i in range(1, future_laps + 1)]
            trend = 'stable'
        
        # Identify critical laps
        critical_laps = [p['lap'] for p in predictions if p['status'] in ['hot', 'critical']]
        
        return {
            'temperature_trend': trend,
            'current_avg_temp': round(float(avg_temps[-1]), 1),
            'current_max_temp': round(float(max_temps[-1]), 1),
            'predictions': predictions,
            'critical_laps_predicted': critical_laps,
            'warning': self._generate_temp_warning(trend, predictions)
        }
    
    def calculate_optimal_compound(self, track_conditions: Dict,
                                  stint_length_target: int,
                                  current_lap: int,
                                  total_laps: int) -> Dict:
        """
        Recommend optimal tyre compound based on conditions and strategy
        
        Returns:
            Compound recommendation with reasoning
        """
        track_temp = track_conditions.get('temperature', 25)
        rainfall = track_conditions.get('rainfall', 0)
        track_evolution = track_conditions.get('track_evolution', 80)
        
        laps_remaining = total_laps - current_lap
        
        # Rain conditions
        if rainfall > 50:
            return {
                'recommended_compound': 'Wet',
                'alternative': 'Intermediate',
                'confidence': 95,
                'reasoning': f'Heavy rain ({rainfall}% rainfall) requires wet tyres'
            }
        elif rainfall > 20:
            return {
                'recommended_compound': 'Intermediate',
                'alternative': 'Soft' if rainfall < 35 else 'Wet',
                'confidence': 85,
                'reasoning': f'Moderate rain conditions ({rainfall}% rainfall)'
            }
        
        # Dry conditions - choose based on temperature and stint length
        compounds = ['Soft', 'Medium', 'Hard']
        scores = {'Soft': 0, 'Medium': 0, 'Hard': 0}
        
        # Temperature scoring
        if track_temp < 20:
            scores['Soft'] += 30
            scores['Medium'] += 15
        elif track_temp < 30:
            scores['Soft'] += 20
            scores['Medium'] += 25
            scores['Hard'] += 15
        else:  # Hot conditions
            scores['Medium'] += 20
            scores['Hard'] += 30
            scores['Soft'] += 5
        
        # Stint length scoring
        if stint_length_target < 15:
            scores['Soft'] += 25
            scores['Medium'] += 10
        elif stint_length_target < 25:
            scores['Soft'] += 10
            scores['Medium'] += 25
            scores['Hard'] += 15
        else:
            scores['Medium'] += 15
            scores['Hard'] += 30
        
        # Laps remaining factor
        if laps_remaining < 15:
            scores['Soft'] += 20  # Sprint to finish
        elif laps_remaining > 30:
            scores['Hard'] += 15  # Need durability
        
        # Track evolution (rubbered in = better grip = can use softer)
        if track_evolution > 85:
            scores['Soft'] += 10
            scores['Medium'] += 5
        
        # Select best compound
        recommended = max(scores, key=scores.get)
        compounds.remove(recommended)
        alternative = max(compounds, key=lambda x: scores[x])
        
        confidence = (scores[recommended] / sum(scores.values())) * 100
        
        reasoning_parts = []
        if track_temp > 30:
            reasoning_parts.append(f"Hot track ({track_temp}°C)")
        if stint_length_target > 25:
            reasoning_parts.append(f"Long stint target ({stint_length_target} laps)")
        if laps_remaining < 15:
            reasoning_parts.append(f"Sprint to finish ({laps_remaining} laps)")
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Balanced conditions"
        
        return {
            'recommended_compound': recommended,
            'alternative': alternative,
            'confidence': round(confidence, 1),
            'reasoning': reasoning,
            'scores': scores
        }
    
    def _calculate_dynamic_wear_rate(self, temperatures: np.ndarray,
                                    track_temp: float, laps: np.ndarray) -> float:
        """Calculate wear rate based on temperature history"""
        if len(temperatures) < 2:
            return self.base_wear_rate
        
        # Recent average temperature
        recent_temp = np.mean(temperatures[-3:]) if len(temperatures) >= 3 else temperatures[-1]
        
        # Base rate adjusted for temperature
        temp_factor = self._get_temp_factor(recent_temp)
        track_factor = 1 + (track_temp - 25) * self.track_temp_factor
        
        wear_rate = self.base_wear_rate * temp_factor * track_factor
        
        return max(0.5, min(8.0, wear_rate))  # Clamp to realistic range
    
    def _get_wear_rate_for_conditions(self, tyre_temp: float, track_temp: float) -> float:
        """Get instantaneous wear rate for given conditions"""
        temp_factor = self._get_temp_factor(tyre_temp)
        track_factor = 1 + (track_temp - 25) * self.track_temp_factor
        return self.base_wear_rate * temp_factor * track_factor
    
    def _get_temp_factor(self, temperature: float) -> float:
        """Calculate wear multiplier based on temperature"""
        if self.optimal_temp_range[0] <= temperature <= self.optimal_temp_range[1]:
            return 1.0  # Optimal
        elif temperature > self.optimal_temp_range[1]:
            # Exponential increase above optimal
            excess_temp = temperature - self.optimal_temp_range[1]
            return 1.0 + (excess_temp * self.temp_wear_multiplier)
        else:
            # Lower wear when cold (but poor grip)
            temp_deficit = self.optimal_temp_range[0] - temperature
            return max(0.7, 1.0 - (temp_deficit * 0.01))
    
    def _get_temp_status(self, temperature: float) -> str:
        """Categorize temperature status"""
        if temperature < 80:
            return 'cold'
        elif temperature < 90:
            return 'cool'
        elif temperature <= 105:
            return 'optimal'
        elif temperature <= 115:
            return 'hot'
        else:
            return 'critical'
    
    def _calculate_model_confidence(self, wear_values: np.ndarray,
                                   laps: np.ndarray, predicted_rate: float) -> float:
        """Calculate confidence in wear rate prediction"""
        if len(wear_values) < 3:
            return 0.4
        
        # Calculate actual wear rate from data
        actual_rate = (wear_values[-1] - wear_values[0]) / (laps[-1] - laps[0])
        
        # Compare predicted vs actual
        error = abs(predicted_rate - actual_rate) / (actual_rate + 0.1)
        
        confidence = max(0.3, 1.0 - error)
        
        # Adjust for sample size
        if len(wear_values) < 5:
            confidence *= 0.7
        elif len(wear_values) >= 10:
            confidence *= 1.1
        
        return min(0.95, confidence)
    
    def _generate_temp_warning(self, trend: str, predictions: List[Dict]) -> str:
        """Generate warning message based on temperature predictions"""
        critical_predictions = [p for p in predictions if p['status'] == 'critical']
        hot_predictions = [p for p in predictions if p['status'] == 'hot']
        
        if critical_predictions:
            first_critical = critical_predictions[0]['lap']
            return f"WARNING: Critical temperature predicted by lap {first_critical}"
        elif hot_predictions and trend == 'rising':
            return f"CAUTION: Temperatures rising, {len(hot_predictions)} laps forecasted hot"
        elif trend == 'rising':
            return "Temperatures increasing - monitor closely"
        elif trend == 'stable' and predictions[0]['status'] == 'optimal':
            return "Temperatures stable in optimal range"
        else:
            return "Temperature conditions normal"
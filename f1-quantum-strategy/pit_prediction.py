"""
Pit Stop Prediction Module
Advanced prediction of optimal pit stop windows using regression and trend analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar

class PitStopPredictor:
    """Predicts optimal pit stop timing based on performance degradation"""
    
    def __init__(self):
        self.performance_threshold = 1.02  # 2% lap time degradation
        self.min_stint_length = 8  # Minimum laps before considering pit
        
    def predict_pit_window(self, telemetry_history: List[Dict], 
                          total_laps: int) -> Dict:
        """
        Predict optimal pit stop window using performance trends
        
        Returns:
            Dictionary with predicted pit lap, window range, and reasoning
        """
        if len(telemetry_history) < 5:
            return {
                'predicted_pit_lap': None,
                'pit_window_start': None,
                'pit_window_end': None,
                'confidence': 0.0,
                'reasoning': 'Insufficient data for prediction'
            }
        
        # Extract time series
        laps = np.array([s['current_lap'] for s in telemetry_history])
        lap_times = np.array([s['lap_time'] for s in telemetry_history])
        tyre_wear = np.array([s['tyre_wear'] for s in telemetry_history])
        
        # Calculate baseline lap time (average of first 3 laps)
        baseline_lap_time = np.mean(lap_times[:min(3, len(lap_times))])
        
        # Fit polynomial to lap time degradation
        if len(laps) >= 3:
            degradation_curve = np.polyfit(laps, lap_times, deg=min(2, len(laps)-1))
            degradation_func = np.poly1d(degradation_curve)
        else:
            # Linear fallback
            degradation_curve = np.polyfit(laps, lap_times, deg=1)
            degradation_func = np.poly1d(degradation_curve)
        
        # Predict when lap time exceeds threshold
        current_lap = laps[-1]
        predicted_laps = np.arange(current_lap + 1, total_laps + 1)
        predicted_times = degradation_func(predicted_laps)
        
        # Find when performance drops below threshold
        threshold_time = baseline_lap_time * self.performance_threshold
        critical_laps = predicted_laps[predicted_times > threshold_time]
        
        if len(critical_laps) > 0:
            predicted_pit_lap = int(critical_laps[0])
        else:
            # Fallback: predict based on tyre wear rate
            predicted_pit_lap = self._predict_from_wear_rate(
                tyre_wear, laps, total_laps
            )
        
        # Calculate pit window (±2 laps from optimal)
        pit_window_start = max(current_lap + 2, predicted_pit_lap - 2)
        pit_window_end = min(total_laps - 3, predicted_pit_lap + 2)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_prediction_confidence(
            telemetry_history, degradation_curve
        )
        
        # Generate reasoning
        current_degradation = ((lap_times[-1] - baseline_lap_time) / baseline_lap_time) * 100
        reasoning = self._generate_pit_reasoning(
            predicted_pit_lap, current_lap, current_degradation, 
            tyre_wear[-1], confidence
        )
        
        return {
            'predicted_pit_lap': predicted_pit_lap,
            'pit_window_start': pit_window_start,
            'pit_window_end': pit_window_end,
            'optimal_window': [pit_window_start, predicted_pit_lap, pit_window_end],
            'confidence': round(confidence * 100, 1),
            'current_degradation_pct': round(current_degradation, 2),
            'laps_until_optimal': predicted_pit_lap - current_lap,
            'reasoning': reasoning,
            'performance_prediction': {
                'baseline_lap_time': round(baseline_lap_time, 3),
                'current_lap_time': round(lap_times[-1], 3),
                'predicted_at_pit': round(float(degradation_func(predicted_pit_lap)), 3)
            }
        }
    
    def evaluate_undercut_overcut(self, telemetry_history: List[Dict],
                                  competitor_data: List[Dict],
                                  current_lap: int) -> Dict:
        """
        Evaluate effectiveness of undercut vs overcut strategies
        
        Returns:
            Analysis of both strategies with expected time gains
        """
        if not competitor_data:
            return {
                'undercut_viable': False,
                'overcut_viable': False,
                'recommendation': 'No competitors to analyze'
            }
        
        our_tyre_age = len(telemetry_history)
        our_wear = telemetry_history[-1]['tyre_wear'] if telemetry_history else 50
        
        strategies = {
            'undercut': {'viable': False, 'expected_gain': 0.0, 'targets': []},
            'overcut': {'viable': False, 'expected_gain': 0.0, 'targets': []}
        }
        
        for competitor in competitor_data:
            comp_tyre_age = competitor.get('tyre_age', our_tyre_age)
            gap = competitor.get('gap', 999)
            
            # Undercut analysis: We pit first
            if our_tyre_age >= self.min_stint_length and comp_tyre_age > our_tyre_age + 3:
                # They're on older tyres, undercut could work
                tyre_advantage = (comp_tyre_age - our_tyre_age) * 0.08  # ~0.08s per lap advantage
                pit_loss = 22.0  # Standard pit stop time loss
                laps_to_catch = int(gap / (tyre_advantage + 0.5)) if tyre_advantage > 0 else 999
                
                if laps_to_catch < 10:  # Viable if we can catch in 10 laps
                    expected_gain = (tyre_advantage * laps_to_catch) - pit_loss
                    if expected_gain > 0:
                        strategies['undercut']['viable'] = True
                        strategies['undercut']['expected_gain'] += expected_gain
                        strategies['undercut']['targets'].append({
                            'car_id': competitor['car_id'],
                            'laps_to_catch': laps_to_catch,
                            'expected_gain': round(expected_gain, 2)
                        })
            
            # Overcut analysis: They pit first, we stay out
            if comp_tyre_age >= self.min_stint_length and our_tyre_age < comp_tyre_age - 2:
                # We have fresher tyres, stay out longer
                tyre_advantage = (comp_tyre_age - our_tyre_age) * 0.05
                extra_laps = min(8, 100 - our_wear) // 3  # How many laps we can extend
                expected_gain = (tyre_advantage * extra_laps) - 1.0  # Track position value
                
                if expected_gain > 0:
                    strategies['overcut']['viable'] = True
                    strategies['overcut']['expected_gain'] += expected_gain
                    strategies['overcut']['targets'].append({
                        'car_id': competitor['car_id'],
                        'extra_laps_out': extra_laps,
                        'expected_gain': round(expected_gain, 2)
                    })
        
        # Determine recommendation
        if strategies['undercut']['viable'] and strategies['overcut']['viable']:
            if strategies['undercut']['expected_gain'] > strategies['overcut']['expected_gain']:
                recommendation = f"UNDERCUT: Pit now for {round(strategies['undercut']['expected_gain'], 1)}s gain"
            else:
                recommendation = f"OVERCUT: Stay out {strategies['overcut']['targets'][0]['extra_laps_out']} more laps"
        elif strategies['undercut']['viable']:
            recommendation = f"UNDERCUT opportunity: Pit within 2 laps"
        elif strategies['overcut']['viable']:
            recommendation = f"OVERCUT: Extend stint to gain track position"
        else:
            recommendation = "Standard pit strategy - no undercut/overcut advantage"
        
        return {
            'undercut': strategies['undercut'],
            'overcut': strategies['overcut'],
            'recommendation': recommendation,
            'analysis_lap': current_lap
        }
    
    def _predict_from_wear_rate(self, tyre_wear: np.ndarray, 
                               laps: np.ndarray, total_laps: int) -> int:
        """Fallback: predict pit lap from tyre wear rate"""
        if len(tyre_wear) < 2:
            return int((laps[-1] + total_laps) / 2)  # Mid-race
        
        # Calculate wear rate
        wear_rate = np.gradient(tyre_wear, laps)
        avg_wear_rate = np.mean(wear_rate[wear_rate > 0]) if len(wear_rate) > 0 else 3.0
        
        # Predict when wear reaches 85% (critical level)
        current_wear = tyre_wear[-1]
        current_lap = laps[-1]
        remaining_wear = 85 - current_wear
        
        if avg_wear_rate > 0:
            laps_remaining = remaining_wear / avg_wear_rate
            predicted_lap = int(current_lap + laps_remaining)
            return min(predicted_lap, total_laps - 3)
        else:
            return int((current_lap + total_laps) / 2)
    
    def _calculate_prediction_confidence(self, telemetry_history: List[Dict],
                                        polynomial: np.ndarray) -> float:
        """Calculate confidence score based on data quality and fit"""
        n_samples = len(telemetry_history)
        
        # Base confidence on sample size
        if n_samples < 5:
            return 0.3
        elif n_samples < 10:
            base_confidence = 0.5
        elif n_samples < 15:
            base_confidence = 0.7
        else:
            base_confidence = 0.85
        
        # Adjust for data consistency (R-squared-like metric)
        lap_times = np.array([s['lap_time'] for s in telemetry_history])
        laps = np.array([s['current_lap'] for s in telemetry_history])
        
        if len(laps) >= 3:
            predicted = np.poly1d(polynomial)(laps)
            ss_res = np.sum((lap_times - predicted) ** 2)
            ss_tot = np.sum((lap_times - np.mean(lap_times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Adjust confidence
            confidence = base_confidence * (0.5 + 0.5 * max(0, r_squared))
        else:
            confidence = base_confidence * 0.6
        
        return min(0.95, max(0.2, confidence))
    
    def _generate_pit_reasoning(self, predicted_lap: int, current_lap: int,
                               degradation_pct: float, tyre_wear: float,
                               confidence: float) -> str:
        """Generate human-readable reasoning for pit prediction"""
        reasons = []
        
        laps_until = predicted_lap - current_lap
        
        if laps_until <= 2:
            reasons.append(f"URGENT: Performance degraded {degradation_pct:.1f}%")
        elif laps_until <= 5:
            reasons.append(f"Pit soon: {laps_until} laps until optimal window")
        else:
            reasons.append(f"Plan ahead: {laps_until} laps to pit window")
        
        if tyre_wear > 70:
            reasons.append(f"High tyre wear ({tyre_wear:.0f}%)")
        
        if confidence > 0.7:
            reasons.append("High confidence prediction")
        elif confidence < 0.5:
            reasons.append("Low confidence - monitor closely")
        
        return " | ".join(reasons)
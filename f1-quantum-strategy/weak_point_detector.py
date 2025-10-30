"""
Weak Point Detection Module
Identifies performance weaknesses and optimization opportunities
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

class WeakPointDetector:
    """Analyzes performance to identify weak points and improvement areas"""
    
    def __init__(self):
        self.sector_names = ["Sector 1", "Sector 2", "Sector 3"]
        self.performance_threshold = 0.015  # 1.5% slower considered weak
        
    def analyze_sector_performance(self, telemetry_history: List[Dict],
                                  competitor_sectors: List[List[float]] = None) -> Dict:
        """
        Deep analysis of sector performance to find weak points
        
        Returns:
            Detailed analysis with specific corner/zone recommendations
        """
        if not telemetry_history or len(telemetry_history) < 3:
            return {
                'weak_sectors': [],
                'strong_sectors': [],
                'consistency_analysis': {},
                'recommendations': []
            }
        
        # Extract sector times across laps
        sector_data = defaultdict(list)
        for sample in telemetry_history:
            if 'sector_times' in sample and len(sample['sector_times']) == 3:
                for i, time in enumerate(sample['sector_times']):
                    sector_data[i].append(time)
        
        # Calculate statistics for each sector
        sector_analysis = []
        for i in range(3):
            if i not in sector_data or len(sector_data[i]) < 3:
                continue
            
            times = np.array(sector_data[i])
            avg_time = np.mean(times)
            std_dev = np.std(times)
            best_time = np.min(times)
            worst_time = np.max(times)
            consistency = (std_dev / avg_time) * 100  # CV as percentage
            
            sector_analysis.append({
                'sector': self.sector_names[i],
                'sector_index': i,
                'average_time': round(avg_time, 3),
                'best_time': round(best_time, 3),
                'worst_time': round(worst_time, 3),
                'std_deviation': round(std_dev, 3),
                'consistency_score': round(100 - consistency, 1),  # Higher is better
                'potential_gain': round(avg_time - best_time, 3),
                'times': times.tolist()
            })
        
        if not sector_analysis:
            return {'error': 'Insufficient sector data'}
        
        # Identify weak and strong sectors
        all_avg_times = [s['average_time'] for s in sector_analysis]
        overall_avg = np.mean(all_avg_times)
        
        weak_sectors = []
        strong_sectors = []
        
        for sector in sector_analysis:
            relative_performance = (sector['average_time'] - overall_avg) / overall_avg
            
            if relative_performance > self.performance_threshold:
                # This sector is significantly slower
                severity = 'critical' if relative_performance > 0.03 else 'high'
                weak_sectors.append({
                    **sector,
                    'severity': severity,
                    'relative_loss': round(relative_performance * 100, 2)
                })
            elif relative_performance < -self.performance_threshold:
                # This sector is significantly faster
                strong_sectors.append({
                    **sector,
                    'relative_advantage': round(abs(relative_performance) * 100, 2)
                })
        
        # Compare with competitors if available
        competitor_comparison = None
        if competitor_sectors and len(competitor_sectors) > 0:
            competitor_comparison = self._compare_with_competitors(
                sector_analysis, competitor_sectors
            )
        
        # Generate specific recommendations
        recommendations = self._generate_sector_recommendations(
            weak_sectors, strong_sectors, sector_analysis
        )
        
        # Consistency analysis
        consistency_analysis = {
            'most_consistent': min(sector_analysis, key=lambda x: x['std_deviation'])['sector'],
            'least_consistent': max(sector_analysis, key=lambda x: x['std_deviation'])['sector'],
            'overall_consistency': round(np.mean([s['consistency_score'] for s in sector_analysis]), 1)
        }
        
        return {
            'weak_sectors': weak_sectors,
            'strong_sectors': strong_sectors,
            'all_sectors': sector_analysis,
            'consistency_analysis': consistency_analysis,
            'competitor_comparison': competitor_comparison,
            'recommendations': recommendations,
            'total_potential_gain_per_lap': round(sum(s['potential_gain'] for s in weak_sectors), 3)
        }
    
    def analyze_lap_time_consistency(self, telemetry_history: List[Dict]) -> Dict:
        """
        Analyze overall lap time consistency and identify anomalies
        
        Returns:
            Consistency metrics and outlier identification
        """
        if len(telemetry_history) < 5:
            return {
                'consistency': 'insufficient_data',
                'outliers': []
            }
        
        lap_times = np.array([s['lap_time'] for s in telemetry_history])
        laps = np.array([s['current_lap'] for s in telemetry_history])
        
        # Calculate statistics
        mean_time = np.mean(lap_times)
        median_time = np.median(lap_times)
        std_dev = np.std(lap_times)
        best_lap = np.min(lap_times)
        worst_lap = np.max(lap_times)
        
        # Coefficient of variation (lower is better)
        cv = (std_dev / mean_time) * 100
        
        # Identify outliers (more than 2 std devs from mean)
        outlier_threshold = 2 * std_dev
        outliers = []
        
        for i, (lap, lap_time) in enumerate(zip(laps, lap_times)):
            deviation = abs(lap_time - mean_time)
            if deviation > outlier_threshold:
                outliers.append({
                    'lap': int(lap),
                    'lap_time': round(lap_time, 3),
                    'deviation_from_mean': round(deviation, 3),
                    'type': 'slow' if lap_time > mean_time else 'fast'
                })
        
        # Determine consistency rating
        if cv < 0.5:
            consistency_rating = 'excellent'
        elif cv < 1.0:
            consistency_rating = 'good'
        elif cv < 2.0:
            consistency_rating = 'average'
        else:
            consistency_rating = 'poor'
        
        # Calculate trend (improving or degrading)
        if len(lap_times) >= 5:
            recent_avg = np.mean(lap_times[-5:])
            early_avg = np.mean(lap_times[:5])
            trend_direction = 'improving' if recent_avg < early_avg else 'degrading'
            trend_magnitude = abs(recent_avg - early_avg)
        else:
            trend_direction = 'stable'
            trend_magnitude = 0.0
        
        return {
            'consistency_rating': consistency_rating,
            'coefficient_variation': round(cv, 3),
            'mean_lap_time': round(mean_time, 3),
            'median_lap_time': round(median_time, 3),
            'std_deviation': round(std_dev, 3),
            'best_lap': round(best_lap, 3),
            'worst_lap': round(worst_lap, 3),
            'lap_time_range': round(worst_lap - best_lap, 3),
            'outliers': outliers,
            'trend': {
                'direction': trend_direction,
                'magnitude': round(trend_magnitude, 3)
            },
            'summary': self._generate_consistency_summary(
                consistency_rating, trend_direction, len(outliers)
            )
        }
    
    def identify_improvement_priorities(self, sector_analysis: Dict,
                                       lap_consistency: Dict,
                                       tyre_condition: float) -> List[Dict]:
        """
        Prioritize improvement areas based on multiple factors
        
        Returns:
            Ordered list of improvement priorities
        """
        priorities = []
        
        # 1. Check for critical weak sectors
        if 'weak_sectors' in sector_analysis:
            for weak in sector_analysis['weak_sectors']:
                if weak['severity'] == 'critical':
                    priorities.append({
                        'priority': 'CRITICAL',
                        'area': weak['sector'],
                        'issue': f"Losing {weak['relative_loss']:.1f}% vs average",
                        'potential_gain': weak['potential_gain'],
                        'action': f"Focus on {weak['sector']} - major time loss",
                        'order': 1
                    })
        
        # 2. Consistency issues
        if lap_consistency.get('consistency_rating') in ['poor', 'average']:
            inconsistent_sector = None
            if 'consistency_analysis' in sector_analysis:
                inconsistent_sector = sector_analysis['consistency_analysis']['least_consistent']
            
            priorities.append({
                'priority': 'HIGH',
                'area': inconsistent_sector or 'Overall lap time',
                'issue': f"Consistency rating: {lap_consistency['consistency_rating']}",
                'potential_gain': lap_consistency['std_deviation'],
                'action': 'Improve consistency - focus on repeatability',
                'order': 2
            })
        
        # 3. Performance degradation
        if lap_consistency.get('trend', {}).get('direction') == 'degrading':
            magnitude = lap_consistency['trend']['magnitude']
            if magnitude > 0.5:
                priorities.append({
                    'priority': 'HIGH',
                    'area': 'Overall performance',
                    'issue': f"Lap times degrading by {magnitude:.3f}s",
                    'potential_gain': magnitude,
                    'action': 'Address tyre degradation or driver fatigue',
                    'order': 2
                })
        
        # 4. Tyre management (if condition is poor)
        if tyre_condition > 70:
            priorities.append({
                'priority': 'MEDIUM',
                'area': 'Tyre management',
                'issue': f"Tyre wear at {tyre_condition:.0f}%",
                'potential_gain': 0.2,  # Estimated
                'action': 'Consider pit stop soon, reduce aggression in corners',
                'order': 3
            })
        
        # 5. Moderate weak sectors
        if 'weak_sectors' in sector_analysis:
            for weak in sector_analysis['weak_sectors']:
                if weak['severity'] != 'critical' and weak not in [p['area'] for p in priorities]:
                    priorities.append({
                        'priority': 'MEDIUM',
                        'area': weak['sector'],
                        'issue': f"Slower than average by {weak['relative_loss']:.1f}%",
                        'potential_gain': weak['potential_gain'],
                        'action': f"Optimize {weak['sector']} when conditions allow",
                        'order': 3
                    })
        
        # Sort by order
        priorities.sort(key=lambda x: x['order'])
        
        return priorities
    
    def _compare_with_competitors(self, our_sectors: List[Dict],
                                 competitor_sectors: List[List[float]]) -> Dict:
        """Compare our sector times with competitors"""
        if not competitor_sectors:
            return None
        
        # Average competitor times per sector
        competitor_averages = []
        for i in range(3):
            sector_times = [comp[i] for comp in competitor_sectors if len(comp) > i]
            if sector_times:
                competitor_averages.append(np.mean(sector_times))
            else:
                competitor_averages.append(None)
        
        comparison = []
        for i, sector in enumerate(our_sectors):
            if i < len(competitor_averages) and competitor_averages[i] is not None:
                our_time = sector['average_time']
                comp_time = competitor_averages[i]
                diff = our_time - comp_time
                diff_pct = (diff / comp_time) * 100
                
                comparison.append({
                    'sector': sector['sector'],
                    'our_time': round(our_time, 3),
                    'competitor_avg': round(comp_time, 3),
                    'difference': round(diff, 3),
                    'difference_pct': round(diff_pct, 2),
                    'status': 'faster' if diff < 0 else 'slower'
                })
        
        return {
            'sector_comparison': comparison,
            'total_lap_difference': round(sum(c['difference'] for c in comparison), 3)
        }
    
    def _generate_sector_recommendations(self, weak_sectors: List[Dict],
                                        strong_sectors: List[Dict],
                                        all_sectors: List[Dict]) -> List[str]:
        """Generate specific actionable recommendations"""
        recommendations = []
        
        # Recommendations for weak sectors
        for weak in weak_sectors:
            sector_idx = weak['sector_index']
            
            if sector_idx == 0:  # Sector 1 - typically high-speed/braking
                recommendations.append(
                    f"{weak['sector']}: Review braking points and turn-in timing. "
                    f"Potential gain: {weak['potential_gain']:.3f}s"
                )
            elif sector_idx == 1:  # Sector 2 - typically technical
                recommendations.append(
                    f"{weak['sector']}: Focus on mid-corner speed and throttle application. "
                    f"Potential gain: {weak['potential_gain']:.3f}s"
                )
            else:  # Sector 3 - typically power/exit
                recommendations.append(
                    f"{weak['sector']}: Optimize corner exits and DRS usage. "
                    f"Potential gain: {weak['potential_gain']:.3f}s"
                )
        
        # Highlight strong sectors to maintain
        if strong_sectors:
            strong_names = [s['sector'] for s in strong_sectors]
            recommendations.append(
                f"Maintain current approach in: {', '.join(strong_names)}"
            )
        
        # Consistency recommendations
        inconsistent = [s for s in all_sectors if s['consistency_score'] < 85]
        if inconsistent:
            recommendations.append(
                f"Improve consistency in {inconsistent[0]['sector']} "
                f"(currently {inconsistent[0]['consistency_score']:.1f}% consistent)"
            )
        
        return recommendations
    
    def _generate_consistency_summary(self, rating: str, trend: str,
                                     outlier_count: int) -> str:
        """Generate human-readable consistency summary"""
        parts = [f"Consistency: {rating}"]
        
        if trend != 'stable':
            parts.append(f"trend: {trend}")
        
        if outlier_count > 0:
            parts.append(f"{outlier_count} outlier lap(s) detected")
        
        return " | ".join(parts)
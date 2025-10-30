"""
Strategy Analyzer
Classical analysis components for overtaking, sector optimization, and risk assessment
"""

from typing import List, Dict
import numpy as np

class StrategyAnalyzer:
    
    def find_overtaking_opportunities(self, our_car, competitors: List, 
                                     drs_zones: List[str]) -> List[Dict]:
        """
        Identify best opportunities to overtake based on:
        - Speed differential
        - Competitor slow zones
        - DRS availability
        - Tyre advantage
        """
        opportunities = []
        
        for competitor in competitors:
            # Only analyze cars directly ahead
            if competitor.position >= our_car.position:
                continue
            
            gap = competitor.gap
            
            # Calculate overtaking probability
            speed_advantage = our_car.speed - competitor.speed
            tyre_advantage = self._calculate_tyre_advantage(
                our_car.tyre_wear, 
                competitor.tyre_age
            )
            
            # Check for vulnerable zones
            vulnerable_zones = self._find_vulnerable_zones(
                competitor.slow_zones,
                drs_zones
            )
            
            # Calculate overtaking probability
            base_probability = 0.3
            if speed_advantage > 5:
                base_probability += 0.2
            if tyre_advantage > 0.2:
                base_probability += 0.15
            if vulnerable_zones:
                base_probability += 0.25
            if gap < 1.0:  # Within DRS range
                base_probability += 0.2
            
            probability = min(base_probability, 0.95)
            
            if probability > 0.4:  # Only suggest if >40% chance
                opportunities.append({
                    "target_car": competitor.car_id,
                    "current_position": competitor.position,
                    "gap": round(gap, 2),
                    "probability": round(probability * 100, 1),
                    "best_zones": vulnerable_zones[:2],  # Top 2 zones
                    "speed_advantage": round(speed_advantage, 1),
                    "tyre_advantage": tyre_advantage > 0.2,
                    "drs_available": gap < 1.0,
                    "recommendation": self._generate_overtake_advice(
                        probability, gap, vulnerable_zones
                    )
                })
        
        # Sort by probability
        opportunities.sort(key=lambda x: x['probability'], reverse=True)
        return opportunities
    
    def optimize_sectors(self, our_slow_sectors: List[str], 
                        competitor_slow_zones: List[List[str]],
                        our_sector_times: List[float]) -> Dict:
        """
        Analyze where we can gain time by comparing our slow sectors
        with where competitors are also slow
        """
        
        # Flatten competitor slow zones
        all_competitor_slow = []
        for zones in competitor_slow_zones:
            all_competitor_slow.extend(zones)
        
        # Count frequency of slow zones
        zone_frequency = {}
        for zone in all_competitor_slow:
            zone_frequency[zone] = zone_frequency.get(zone, 0) + 1
        
        # Analyze our sectors
        sector_analysis = []
        for i, sector_time in enumerate(our_sector_times):
            sector_name = f"Sector {i + 1}"
            
            # Check if we're slow here
            is_our_weakness = sector_name.lower().replace(" ", "_") in [
                s.lower().replace(" ", "_") for s in our_slow_sectors
            ]
            
            # Check if competitors are also slow here
            competitor_struggle = zone_frequency.get(sector_name, 0)
            
            # Priority calculation
            if is_our_weakness and competitor_struggle == 0:
                priority = "CRITICAL"  # We're slow, they're not - big opportunity
                potential_gain = 0.4
            elif is_our_weakness and competitor_struggle > 0:
                priority = "HIGH"  # Everyone struggles here
                potential_gain = 0.2
            elif not is_our_weakness and competitor_struggle > 2:
                priority = "MEDIUM"  # We're good, others struggle - maintain advantage
                potential_gain = 0.0
            else:
                priority = "LOW"
                potential_gain = 0.1
            
            sector_analysis.append({
                "sector": sector_name,
                "current_time": round(sector_time, 3),
                "priority": priority,
                "potential_gain": round(potential_gain, 2),
                "our_weakness": is_our_weakness,
                "competitors_struggle": competitor_struggle,
                "advice": self._generate_sector_advice(
                    priority, is_our_weakness, competitor_struggle
                )
            })
        
        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sector_analysis.sort(key=lambda x: priority_order[x['priority']])
        
        # Calculate total potential gain
        total_gain = sum(s['potential_gain'] for s in sector_analysis)
        
        return {
            "sectors": sector_analysis,
            "total_potential_gain": round(total_gain, 2),
            "focus_sector": sector_analysis[0]['sector'] if sector_analysis else None,
            "summary": self._generate_sector_summary(sector_analysis)
        }
    
    def calculate_expected_gain(self, pit_recommendation: Dict,
                               pace_strategy: Dict,
                               sector_optimization: Dict) -> float:
        """
        Calculate total expected time gain from all strategic recommendations
        """
        total_gain = 0.0
        
        # Pit stop impact
        if pit_recommendation.get('expected_time_impact'):
            total_gain += pit_recommendation['expected_time_impact']
        
        # Pace strategy impact
        pace_mode = pace_strategy.get('pace_mode', 'BALANCED')
        if pace_mode == 'ATTACK':
            total_gain += -2.0  # Gain 2 seconds
        elif pace_mode == 'PUSH':
            total_gain += -1.0
        
        # Sector optimization impact
        total_gain += sector_optimization.get('total_potential_gain', 0) * -1
        
        return round(total_gain, 2)
    
    def assess_risk(self, pit_recommendation: Dict, track_conditions,
                   current_position: int) -> str:
        """
        Assess overall risk level of the strategy
        """
        risk_factors = []
        risk_score = 0
        
        # Weather risk
        if track_conditions.rainfall > 30:
            risk_factors.append("Rain conditions")
            risk_score += 2
        
        # Pit stop risk
        pit_confidence = pit_recommendation.get('confidence', 100)
        if pit_confidence < 60:
            risk_factors.append("Uncertain pit timing")
            risk_score += 1
        
        # Position risk
        if current_position <= 3:
            risk_factors.append("Leading position - conservative approach recommended")
            risk_score += 1
        
        # Tyre temperature risk
        recommendation = pit_recommendation.get('recommendation', '')
        if 'URGENT' in recommendation:
            risk_factors.append("Critical tyre condition")
            risk_score += 2
        
        # Determine risk level
        if risk_score >= 4:
            level = "HIGH"
        elif risk_score >= 2:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        risk_description = f"{level} RISK"
        if risk_factors:
            risk_description += f": {', '.join(risk_factors)}"
        
        return risk_description
    
    def _calculate_tyre_advantage(self, our_wear: float, 
                                 competitor_tyre_age: int) -> float:
        """
        Calculate tyre advantage/disadvantage
        Returns positive if we have advantage
        """
        # Estimate competitor wear (assume ~3% per lap)
        estimated_competitor_wear = min(competitor_tyre_age * 3, 100)
        
        advantage = (estimated_competitor_wear - our_wear) / 100
        return advantage
    
    def _find_vulnerable_zones(self, competitor_slow_zones: List[str],
                              drs_zones: List[str]) -> List[str]:
        """
        Find zones where overtaking is most likely
        """
        vulnerable = []
        
        # DRS zones where competitor is slow
        for zone in drs_zones:
            if any(zone.lower() in slow.lower() for slow in competitor_slow_zones):
                vulnerable.append(zone)
        
        # Their slow zones even without DRS
        for slow_zone in competitor_slow_zones:
            if slow_zone not in vulnerable:
                vulnerable.append(slow_zone)
        
        return vulnerable
    
    def _generate_overtake_advice(self, probability: float, gap: float,
                                 zones: List[str]) -> str:
        """
        Generate tactical overtaking advice
        """
        if probability > 0.7:
            if gap < 1.0:
                return f"High chance! Attack in {zones[0] if zones else 'next DRS zone'} with DRS"
            else:
                return f"Build pressure, close gap to <1.0s for DRS"
        elif probability > 0.5:
            return f"Moderate chance - wait for mistake in {zones[0] if zones else 'slow corners'}"
        else:
            return "Monitor situation, consider alternate strategy"
    
    def _generate_sector_advice(self, priority: str, our_weakness: bool,
                               competitor_struggle: int) -> str:
        """
        Generate specific sector improvement advice
        """
        if priority == "CRITICAL":
            return "FOCUS HERE: Major time loss vs competitors. Review braking points and racing line"
        elif priority == "HIGH":
            return "Difficult sector for all - small improvements yield big gains"
        elif priority == "MEDIUM":
            return "Maintain current advantage - don't take unnecessary risks"
        else:
            return "Satisfactory performance - monitor only"
    
    def _generate_sector_summary(self, sector_analysis: List[Dict]) -> str:
        """
        Generate overall sector strategy summary
        """
        critical = sum(1 for s in sector_analysis if s['priority'] == 'CRITICAL')
        high = sum(1 for s in sector_analysis if s['priority'] == 'HIGH')
        
        if critical > 0:
            focus = [s['sector'] for s in sector_analysis if s['priority'] == 'CRITICAL']
            return f"Critical improvement needed in: {', '.join(focus)}"
        elif high > 0:
            focus = [s['sector'] for s in sector_analysis if s['priority'] == 'HIGH']
            return f"Focus on: {', '.join(focus)}"
        else:
            return "Sector times competitive - maintain current approach"
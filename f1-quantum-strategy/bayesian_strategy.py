"""
Bayesian Probability Updater for Race Strategy
Updates confidence as race progresses and new evidence arrives
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import beta
from dataclasses import dataclass

@dataclass
class StrategyBelief:
    """Bayesian belief about a strategy's success probability"""
    alpha: float  # Success count + prior
    beta_param: float  # Failure count + prior
    last_update_lap: int
    evidence_count: int
    
    def get_mean_probability(self) -> float:
        """Expected value of Beta distribution"""
        return self.alpha / (self.alpha + self.beta_param)
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Credible interval for probability"""
        dist = beta(self.alpha, self.beta_param)
        lower = dist.ppf((1 - confidence) / 2)
        upper = dist.ppf((1 + confidence) / 2)
        return (lower, upper)
    
    def get_certainty(self) -> float:
        """How certain are we? (0-1, higher = more certain)"""
        # Based on variance of Beta distribution
        variance = (self.alpha * self.beta_param) / (
            (self.alpha + self.beta_param)**2 * (self.alpha + self.beta_param + 1)
        )
        certainty = 1 - (variance * 10)  # Scale to 0-1
        return max(0, min(1, certainty))


class BayesianStrategyUpdater:
    """
    Updates strategy probabilities using Bayesian inference
    
    As race progresses, we gather evidence:
    - Tyre degradation rate observed → update pit timing belief
    - Opponent behavior observed → update overtake probability
    - Track conditions evolving → update strategy viability
    """
    
    def __init__(self):
        # Initialize beliefs for each strategy type
        self.beliefs: Dict[str, StrategyBelief] = {
            'pit_early': StrategyBelief(5, 5, 0, 0),    # Neutral prior (50/50)
            'pit_mid': StrategyBelief(7, 3, 0, 0),      # Slightly optimistic prior
            'pit_late': StrategyBelief(4, 6, 0, 0),     # Slightly pessimistic
            'overtake': StrategyBelief(3, 7, 0, 0),     # Conservative prior (risky)
            'hold_position': StrategyBelief(8, 2, 0, 0), # Safe strategy
            'push_pace': StrategyBelief(5, 5, 0, 0),
            'conserve': StrategyBelief(6, 4, 0, 0)
        }
        
        # Track evidence history
        self.evidence_log: List[Dict] = []
    
    def update_with_evidence(self, strategy: str, observation: Dict, 
                            current_lap: int) -> Dict:
        """
        Update belief about strategy using new evidence
        
        Args:
            strategy: Strategy name (e.g., 'pit_early')
            observation: Evidence dict with 'favorable', 'strength'
            current_lap: Current race lap
            
        Returns:
            Updated probability and confidence metrics
        """
        if strategy not in self.beliefs:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        belief = self.beliefs[strategy]
        
        # Extract evidence
        is_favorable = observation.get('favorable', True)
        strength = observation.get('strength', 1.0)  # 0-1 scale
        
        # Update belief using Bayesian update
        if is_favorable:
            # Add to success count
            belief.alpha += strength
        else:
            # Add to failure count
            belief.beta_param += strength
        
        belief.last_update_lap = current_lap
        belief.evidence_count += 1
        
        # Log evidence
        self.evidence_log.append({
            'lap': current_lap,
            'strategy': strategy,
            'observation': observation,
            'new_probability': belief.get_mean_probability()
        })
        
        # Return updated assessment
        return {
            'strategy': strategy,
            'probability': belief.get_mean_probability(),
            'confidence_interval': belief.get_confidence_interval(),
            'certainty': belief.get_certainty(),
            'evidence_count': belief.evidence_count,
            'last_updated': current_lap
        }
    
    def compare_strategies(self, strategies: List[str]) -> Dict:
        """
        Compare multiple strategies and return ranking
        
        Uses Thompson Sampling for exploration/exploitation
        """
        samples = {}
        
        for strategy in strategies:
            if strategy in self.beliefs:
                belief = self.beliefs[strategy]
                # Sample from Beta distribution
                sample = np.random.beta(belief.alpha, belief.beta_param)
                samples[strategy] = sample
        
        # Rank strategies
        ranked = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'recommended': ranked[0][0],
            'ranking': ranked,
            'probabilities': {
                strategy: self.beliefs[strategy].get_mean_probability()
                for strategy in strategies if strategy in self.beliefs
            }
        }
    
    def update_pit_timing_belief(self, telemetry_data: Dict, 
                                 current_lap: int) -> Dict:
        """
        Update pit timing beliefs based on tyre degradation evidence
        
        Evidence sources:
        - Actual wear rate vs predicted
        - Temperature trends
        - Lap time degradation
        """
        tyre_wear = telemetry_data.get('tyre_wear', 50)
        tyre_temp = telemetry_data.get('avg_tyre_temp', 95)
        wear_rate = telemetry_data.get('wear_rate', 2.5)
        lap_time_delta = telemetry_data.get('lap_time_delta', 0)
        
        updates = {}
        
        # Update pit_early belief
        if tyre_wear > 60 or tyre_temp > 110:
            # High wear or temp → pit early looks good
            updates['pit_early'] = self.update_with_evidence(
                'pit_early',
                {'favorable': True, 'strength': 0.5},
                current_lap
            )
        elif tyre_wear < 40 and tyre_temp < 95:
            # Low wear and temp → pit early not needed
            updates['pit_early'] = self.update_with_evidence(
                'pit_early',
                {'favorable': False, 'strength': 0.3},
                current_lap
            )
        
        # Update pit_mid belief
        if 40 <= tyre_wear <= 70:
            # In mid-range → pit_mid strategy reasonable
            updates['pit_mid'] = self.update_with_evidence(
                'pit_mid',
                {'favorable': True, 'strength': 0.4},
                current_lap
            )
        
        # Update pit_late belief
        if tyre_wear < 50 and wear_rate < 2.0:
            # Slow wear → can extend stint
            updates['pit_late'] = self.update_with_evidence(
                'pit_late',
                {'favorable': True, 'strength': 0.6},
                current_lap
            )
        elif lap_time_delta > 0.5:
            # Lap times degrading → can't wait longer
            updates['pit_late'] = self.update_with_evidence(
                'pit_late',
                {'favorable': False, 'strength': 0.7},
                current_lap
            )
        
        return updates
    
    def update_overtake_belief(self, race_state: Dict, current_lap: int) -> Dict:
        """
        Update overtake belief based on gap, speed, and tyre advantage
        
        Evidence sources:
        - Gap to car ahead
        - Speed differential
        - Tyre age differential
        - DRS availability
        """
        gap_ahead = race_state.get('gap_ahead', 5.0)
        speed_advantage = race_state.get('speed_advantage', 0)
        our_tyre_age = race_state.get('our_tyre_age', 10)
        rival_tyre_age = race_state.get('rival_tyre_age', 10)
        drs_available = race_state.get('drs_available', False)
        
        # Calculate evidence strength
        evidence_factors = []
        
        # Gap factor
        if gap_ahead < 1.0:
            evidence_factors.append(('gap', True, 0.4))
        elif gap_ahead > 3.0:
            evidence_factors.append(('gap', False, 0.5))
        
        # Speed factor
        if speed_advantage > 5:
            evidence_factors.append(('speed', True, 0.3))
        elif speed_advantage < -5:
            evidence_factors.append(('speed', False, 0.4))
        
        # Tyre advantage
        tyre_delta = rival_tyre_age - our_tyre_age
        if tyre_delta > 5:
            evidence_factors.append(('tyre', True, 0.5))
        elif tyre_delta < -5:
            evidence_factors.append(('tyre', False, 0.4))
        
        # DRS
        if drs_available:
            evidence_factors.append(('drs', True, 0.6))
        
        # Aggregate evidence
        favorable_strength = sum(s for _, f, s in evidence_factors if f)
        unfavorable_strength = sum(s for _, f, s in evidence_factors if not f)
        
        if favorable_strength > unfavorable_strength:
            return self.update_with_evidence(
                'overtake',
                {'favorable': True, 'strength': favorable_strength},
                current_lap
            )
        elif unfavorable_strength > favorable_strength:
            return self.update_with_evidence(
                'overtake',
                {'favorable': False, 'strength': unfavorable_strength},
                current_lap
            )
        else:
            return self.get_current_belief('overtake')
    
    def update_pace_belief(self, tyre_state: Dict, current_lap: int) -> Dict:
        """
        Update push_pace vs conserve beliefs based on tyre condition
        """
        tyre_wear = tyre_state.get('tyre_wear', 50)
        tyre_temp = tyre_state.get('tyre_temp', 95)
        laps_on_stint = tyre_state.get('laps_on_stint', 10)
        
        updates = {}
        
        # Push pace evidence
        if tyre_wear < 40 and tyre_temp < 100:
            # Good tyre condition → can push
            updates['push_pace'] = self.update_with_evidence(
                'push_pace',
                {'favorable': True, 'strength': 0.5},
                current_lap
            )
        elif tyre_wear > 70 or tyre_temp > 110:
            # Poor condition → shouldn't push
            updates['push_pace'] = self.update_with_evidence(
                'push_pace',
                {'favorable': False, 'strength': 0.6},
                current_lap
            )
        
        # Conserve evidence
        if laps_on_stint > 20 and tyre_wear > 60:
            # Long stint, high wear → should conserve
            updates['conserve'] = self.update_with_evidence(
                'conserve',
                {'favorable': True, 'strength': 0.6},
                current_lap
            )
        elif tyre_wear < 30:
            # Fresh tyres → don't need to conserve
            updates['conserve'] = self.update_with_evidence(
                'conserve',
                {'favorable': False, 'strength': 0.4},
                current_lap
            )
        
        return updates
    
    def get_current_belief(self, strategy: str) -> Dict:
        """Get current belief without updating"""
        if strategy not in self.beliefs:
            return {}
        
        belief = self.beliefs[strategy]
        return {
            'strategy': strategy,
            'probability': belief.get_mean_probability(),
            'confidence_interval': belief.get_confidence_interval(),
            'certainty': belief.get_certainty(),
            'evidence_count': belief.evidence_count
        }
    
    def get_all_beliefs(self) -> Dict:
        """Get all current strategy beliefs"""
        return {
            strategy: self.get_current_belief(strategy)
            for strategy in self.beliefs.keys()
        }
    
    def reset_beliefs(self):
        """Reset all beliefs to priors (e.g., for new race)"""
        self.__init__()
    
    def get_decision_recommendation(self, current_lap: int, 
                                   race_state: Dict) -> Dict:
        """
        Main decision function using Bayesian beliefs
        
        Returns comprehensive recommendation with probabilities
        """
        # Update all beliefs with current evidence
        self.update_pit_timing_belief(race_state, current_lap)
        self.update_overtake_belief(race_state, current_lap)
        self.update_pace_belief(race_state, current_lap)
        
        # Get all pit timing options
        pit_strategies = ['pit_early', 'pit_mid', 'pit_late']
        pit_comparison = self.compare_strategies(pit_strategies)
        
        # Get overtake assessment
        overtake_belief = self.get_current_belief('overtake')
        
        # Get pace strategy
        pace_strategies = ['push_pace', 'conserve', 'hold_position']
        pace_comparison = self.compare_strategies(pace_strategies)
        
        # Combine into overall recommendation
        all_beliefs = self.get_all_beliefs()
        
        # Find highest probability action
        best_action = max(
            all_beliefs.items(),
            key=lambda x: x[1].get('probability', 0)
        )
        
        return {
            'recommended_action': best_action[0],
            'probability': best_action[1]['probability'],
            'certainty': best_action[1]['certainty'],
            'confidence_interval': best_action[1]['confidence_interval'],
            'pit_strategy': pit_comparison,
            'overtake_assessment': overtake_belief,
            'pace_strategy': pace_comparison,
            'all_probabilities': {
                k: v['probability'] for k, v in all_beliefs.items()
            },
            'evidence_summary': {
                k: v['evidence_count'] for k, v in all_beliefs.items()
            }
        }


# Example usage
if __name__ == '__main__':
    updater = BayesianStrategyUpdater()
    
    # Simulate race progression
    print("🏎️ Simulating Bayesian strategy updates across race\n")
    
    for lap in range(1, 31, 5):
        print(f"📍 LAP {lap}")
        
        # Simulate telemetry
        race_state = {
            'tyre_wear': lap * 2.5,
            'avg_tyre_temp': 95 + lap * 0.5,
            'wear_rate': 2.5,
            'lap_time_delta': lap * 0.02,
            'gap_ahead': max(0.5, 3.0 - lap * 0.1),
            'speed_advantage': 5 if lap > 15 else 0,
            'our_tyre_age': lap,
            'rival_tyre_age': lap + 5,
            'drs_available': lap % 3 == 0,
            'laps_on_stint': lap
        }
        
        # Get recommendation
        recommendation = updater.get_decision_recommendation(lap, race_state)
        
        print(f"  Best action: {recommendation['recommended_action']}")
        print(f"  Probability: {recommendation['probability']:.1%}")
        print(f"  Certainty: {recommendation['certainty']:.1%}")
        print(f"  CI: [{recommendation['confidence_interval'][0]:.2f}, {recommendation['confidence_interval'][1]:.2f}]")
        print()
    
    print("✅ Bayesian updates complete!")
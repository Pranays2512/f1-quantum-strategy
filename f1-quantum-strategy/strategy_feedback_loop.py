"""
Integrated Strategy Feedback Loop
Connects all modules into a real-time decision system
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class StrategyAction(Enum):
    OVERTAKE = "OVERTAKE"
    PIT_NOW = "PIT_NOW"
    PIT_SOON = "PIT_SOON"
    HOLD_POSITION = "HOLD_POSITION"
    DEFEND = "DEFEND"
    PUSH_PACE = "PUSH_PACE"
    CONSERVE = "CONSERVE"

@dataclass
class RaceState:
    """Complete race state snapshot"""
    lap: int
    position: int
    tyre_wear: float
    tyre_temps: Dict[str, float]
    fuel: float
    gap_ahead: float
    gap_behind: float
    sector_times: List[float]
    competitors: List[Dict]
    track_conditions: Dict
    total_laps: int

@dataclass
class StrategyDecision:
    """Output of strategy decision process"""
    primary_action: StrategyAction
    confidence: float
    expected_position: float
    expected_time_delta: float
    risk_level: str
    reasoning: List[str]
    alternative_actions: List[Tuple[StrategyAction, float]]
    should_execute_now: bool

class IntegratedStrategyEngine:
    """
    Unified decision engine that integrates:
    - Weak point detection → overtake probability
    - Tyre modeling → pit timing
    - Strategy analyzer → multi-path comparison
    - Quantum optimization → decision selection
    """
    
    def __init__(self, weak_point_detector, tyre_model, pit_predictor, 
                 strategy_analyzer, quantum_engine):
        self.weak_detector = weak_point_detector
        self.tyre_model = tyre_model
        self.pit_predictor = pit_predictor
        self.strategy_analyzer = strategy_analyzer
        self.quantum_engine = quantum_engine
        
        # State tracking for feedback
        self.race_history: List[RaceState] = []
        self.decision_history: List[StrategyDecision] = []
        self.confidence_trends: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.action_outcomes: Dict[StrategyAction, List[float]] = {
            action: [] for action in StrategyAction
        }
    
    def decide_optimal_strategy(self, race_state: RaceState, 
                               telemetry_history: List[Dict]) -> StrategyDecision:
        """
        Main decision loop: evaluate all options and pick best action
        
        Returns: Optimal strategy with confidence and reasoning
        """
        # Store state for feedback
        self.race_history.append(race_state)
        
        # STEP 1: Analyze all strategic options
        strategy_options = self._generate_strategy_options(race_state)
        
        # STEP 2: Evaluate each option with integrated modules
        evaluated_options = []
        
        for option in strategy_options:
            evaluation = self._evaluate_strategy_option(
                option, race_state, telemetry_history
            )
            evaluated_options.append(evaluation)
        
        # STEP 3: Compare paths using expected value
        best_option = self._select_best_strategy(evaluated_options)
        
        # STEP 4: Create decision with feedback-adjusted confidence
        decision = self._create_decision(best_option, evaluated_options, race_state)
        
        # STEP 5: Update feedback loop
        self._update_feedback_loop(decision, race_state)
        
        return decision
    
    def _generate_strategy_options(self, race_state: RaceState) -> List[Dict]:
        """Generate all viable strategy options for current state"""
        options = []
        
        # Option 1: Overtake (if competitor ahead)
        if race_state.gap_ahead < 3.0 and race_state.position > 1:
            options.append({
                'action': StrategyAction.OVERTAKE,
                'target_lap': race_state.lap + 1,
                'parameters': {'aggressive': True}
            })
        
        # Option 2: Pit now
        if race_state.tyre_wear > 40:
            options.append({
                'action': StrategyAction.PIT_NOW,
                'target_lap': race_state.lap + 1,
                'parameters': {'compound': 'medium'}
            })
        
        # Option 3: Pit in 3-5 laps
        if race_state.tyre_wear > 30:
            for pit_lap in [race_state.lap + 3, race_state.lap + 5]:
                if pit_lap < race_state.total_laps - 3:
                    options.append({
                        'action': StrategyAction.PIT_SOON,
                        'target_lap': pit_lap,
                        'parameters': {'compound': 'medium'}
                    })
        
        # Option 4: Hold position and conserve
        options.append({
            'action': StrategyAction.HOLD_POSITION,
            'target_lap': race_state.lap,
            'parameters': {'pace': 'conservative'}
        })
        
        # Option 5: Push pace (if tyres good)
        if race_state.tyre_wear < 60:
            options.append({
                'action': StrategyAction.PUSH_PACE,
                'target_lap': race_state.lap,
                'parameters': {'pace': 'aggressive'}
            })
        
        # Option 6: Defend (if competitor behind)
        if race_state.gap_behind < 1.5:
            options.append({
                'action': StrategyAction.DEFEND,
                'target_lap': race_state.lap,
                'parameters': {'defensive_lines': True}
            })
        
        return options
    
    def _evaluate_strategy_option(self, option: Dict, race_state: RaceState,
                                  history: List[Dict]) -> Dict:
        """
        Evaluate single strategy option using all integrated modules
        
        Returns: Option with EV, confidence, risk assessment
        """
        action = option['action']
        
        # Initialize evaluation
        evaluation = {
            'action': action,
            'option_params': option,
            'expected_value': 0.0,
            'confidence': 0.5,
            'risk_score': 0.5,
            'time_impact': 0.0,
            'position_impact': 0.0,
            'component_scores': {}
        }
        
        if action == StrategyAction.OVERTAKE:
            # Use weak point detector to enhance overtake probability
            overtake_eval = self._evaluate_overtake_with_weakpoints(
                race_state, history
            )
            evaluation.update(overtake_eval)
            
        elif action in [StrategyAction.PIT_NOW, StrategyAction.PIT_SOON]:
            # Use tyre model + pit predictor for timing optimization
            pit_eval = self._evaluate_pit_with_tyre_model(
                option['target_lap'], race_state, history
            )
            evaluation.update(pit_eval)
            
        elif action == StrategyAction.PUSH_PACE:
            # Use tyre model to assess degradation risk
            pace_eval = self._evaluate_pace_strategy(race_state, history)
            evaluation.update(pace_eval)
            
        elif action == StrategyAction.HOLD_POSITION:
            # Baseline: maintain current pace
            hold_eval = self._evaluate_hold_strategy(race_state)
            evaluation.update(hold_eval)
            
        elif action == StrategyAction.DEFEND:
            # Use competitor analysis for defensive strategy
            defend_eval = self._evaluate_defensive_strategy(race_state, history)
            evaluation.update(defend_eval)
        
        # Apply feedback loop adjustments
        evaluation = self._apply_feedback_adjustments(evaluation, action)
        
        return evaluation
    
    def _evaluate_overtake_with_weakpoints(self, race_state: RaceState,
                                          history: List[Dict]) -> Dict:
        """
        Enhanced overtake evaluation using weak point detection
        
        Integration: weak_point_detector → overtake probability
        """
        # Analyze opponent weak sectors
        opponent_analysis = self.weak_detector.analyze_sector_performance(
            history, [[28.0, 32.5, 28.5]]  # Opponent reference times
        )
        
        weak_sectors = opponent_analysis.get('weak_sectors', [])
        
        # Base overtake probability from speed/gap
        base_prob = 0.3
        gap = race_state.gap_ahead
        
        if gap < 1.0:
            base_prob = 0.6  # DRS range
        elif gap < 2.0:
            base_prob = 0.45
        
        # ENHANCEMENT: Increase probability if opponent weak in upcoming sector
        current_sector = race_state.lap % 3
        for weak in weak_sectors:
            if weak['sector_index'] == current_sector:
                base_prob += 0.2  # Significant boost
                break
        
        # Calculate expected value
        # Success: gain 1 position, avg time gain ~3s
        # Failure: lose time in attempt, risk position to car behind
        success_value = 3.0  # seconds gained
        failure_cost = -1.5   # seconds lost
        
        expected_value = (base_prob * success_value) + ((1 - base_prob) * failure_cost)
        
        # Risk assessment
        risk_score = 0.3 if gap < 1.0 else 0.6  # Less risk with DRS
        
        return {
            'expected_value': expected_value,
            'confidence': base_prob,
            'risk_score': risk_score,
            'time_impact': expected_value,
            'position_impact': -base_prob,  # Negative = better position
            'component_scores': {
                'base_probability': base_prob,
                'weak_sector_boost': len(weak_sectors) > 0,
                'drs_available': gap < 1.0
            }
        }
    
    def _evaluate_pit_with_tyre_model(self, pit_lap: int, race_state: RaceState,
                                     history: List[Dict]) -> Dict:
        """
        Enhanced pit evaluation using tyre degradation modeling
        
        Integration: tyre_model + pit_predictor → optimal timing
        """
        # Predict tyre life remaining
        tyre_prediction = self.tyre_model.predict_tyre_life(
            history, 
            race_state.track_conditions.get('temperature', 25),
            race_state.total_laps
        )
        
        # Predict optimal pit window
        pit_window = self.pit_predictor.predict_pit_window(
            history, race_state.total_laps
        )
        
        # Calculate if this pit lap is in optimal window
        optimal_lap = pit_window.get('predicted_pit_lap', pit_lap)
        if optimal_lap is None:
            optimal_lap = pit_lap  # Fallback if prediction returns None
        window_start = pit_window.get('pit_window_start')
        window_end = pit_window.get('pit_window_end')
        if window_start is None:
            window_start = optimal_lap - 2
        if window_end is None:
            window_end = optimal_lap + 2
        
        in_optimal_window = window_start <= pit_lap <= window_end
        
        # Time impact calculation
        laps_before_pit = pit_lap - race_state.lap
        laps_after_pit = race_state.total_laps - pit_lap
        
        # Degradation cost if we wait
        wear_rate = tyre_prediction.get('current_wear_rate', 2.5)
        degradation_cost = laps_before_pit * (wear_rate * 0.1)  # ~0.1s per % wear
        
        # Pit stop time loss
        pit_loss = 22.0
        
        # Gain from fresh tyres
        tyre_gain = laps_after_pit * 0.3  # ~0.3s per lap advantage
        
        net_time_impact = -degradation_cost - pit_loss + tyre_gain
        
        # Confidence from model
        confidence = pit_window.get('confidence', 50) / 100
        if in_optimal_window:
            confidence *= 1.2  # Boost if in window
        
        # Expected value
        expected_value = net_time_impact / 10  # Normalize
        
        # Risk score (higher risk if pitting too early)
        if race_state.tyre_wear < 50:
            risk_score = 0.7  # Early pit = risky
        elif race_state.tyre_wear > 80:
            risk_score = 0.3  # Late pit = must do it
        else:
            risk_score = 0.5
        
        return {
            'expected_value': expected_value,
            'confidence': min(0.95, confidence),
            'risk_score': risk_score,
            'time_impact': net_time_impact,
            'position_impact': 0.5,  # Typically lose ~0.5 positions
            'component_scores': {
                'in_optimal_window': in_optimal_window,
                'degradation_cost': degradation_cost,
                'fresh_tyre_gain': tyre_gain,
                'predicted_failure_lap': tyre_prediction.get('predicted_failure_lap')
            }
        }
    
    def _evaluate_pace_strategy(self, race_state: RaceState,
                               history: List[Dict]) -> Dict:
        """Evaluate pushing pace vs conserving tyres"""
        # Check if tyres can handle aggressive pace
        temp_forecast = self.tyre_model.predict_temperature_evolution(
            history, future_laps=5
        )
        
        will_overheat = any(
            p['status'] in ['hot', 'critical'] 
            for p in temp_forecast.get('predictions', [])
        )
        
        if will_overheat:
            # Can't push - tyres will overheat
            return {
                'expected_value': -2.0,
                'confidence': 0.8,
                'risk_score': 0.8,
                'time_impact': -1.0,
                'position_impact': 0.2,
                'component_scores': {'overheat_risk': True}
            }
        
        # Can push safely
        laps_remaining = race_state.total_laps - race_state.lap
        time_gain_per_lap = 0.4  # Aggressive pace
        total_gain = min(10, laps_remaining) * time_gain_per_lap
        
        return {
            'expected_value': total_gain / 10,
            'confidence': 0.7,
            'risk_score': 0.5,
            'time_impact': -total_gain,
            'position_impact': -0.3,
            'component_scores': {'can_push': True}
        }
    
    def _evaluate_hold_strategy(self, race_state: RaceState) -> Dict:
        """Baseline: hold position"""
        return {
            'expected_value': 0.0,
            'confidence': 0.9,
            'risk_score': 0.2,
            'time_impact': 0.0,
            'position_impact': 0.0,
            'component_scores': {'baseline': True}
        }
    
    def _evaluate_defensive_strategy(self, race_state: RaceState,
                                    history: List[Dict]) -> Dict:
        """Evaluate defensive driving"""
        gap_behind = race_state.gap_behind
        
        if gap_behind > 2.0:
            # No real threat
            return {'expected_value': -1.0, 'confidence': 0.5, 'risk_score': 0.3}
        
        # Real threat - defend
        return {
            'expected_value': 1.0,  # Preserving position is valuable
            'confidence': 0.7,
            'risk_score': 0.4,
            'time_impact': 0.5,  # Slight time loss from defensive lines
            'position_impact': 0.0,  # Maintain position
            'component_scores': {'threat_level': 'high'}
        }
    
    def _select_best_strategy(self, evaluated_options: List[Dict]) -> Dict:
        """
        Select best strategy using multi-objective optimization
        
        Considers: EV, confidence, risk, time impact
        """
        # Score each option
        scored_options = []
        
        for option in evaluated_options:
            # Weighted scoring
            score = (
                option['expected_value'] * 0.4 +
                option['confidence'] * 0.3 -
                option['risk_score'] * 0.2 +
                (-option['position_impact']) * 0.1  # Better position = negative value
            )
            
            scored_options.append({
                **option,
                'total_score': score
            })
        
        # Select highest score
        best = max(scored_options, key=lambda x: x['total_score'])
        
        return best
    
    def _create_decision(self, best_option: Dict, all_options: List[Dict],
                        race_state: RaceState) -> StrategyDecision:
        """Package decision with alternatives and reasoning"""
        # Generate reasoning
        reasoning = []
        reasoning.append(
            f"Selected {best_option['action'].value} with {best_option['confidence']*100:.0f}% confidence"
        )
        
        if 'component_scores' in best_option:
            for key, value in best_option['component_scores'].items():
                reasoning.append(f"{key}: {value}")
        
        # Sort alternatives
        alternatives = sorted(
            [(opt['action'], opt.get('total_score', 0)) for opt in all_options],
            key=lambda x: x[1],
            reverse=True
        )[1:4]  # Top 3 alternatives
        
        # Determine if should execute immediately
        should_execute = (
            best_option['action'] in [StrategyAction.OVERTAKE, StrategyAction.PIT_NOW, StrategyAction.DEFEND] or
            best_option['confidence'] > 0.8
        )
        
        # Risk level
        if best_option['risk_score'] < 0.3:
            risk_level = 'LOW'
        elif best_option['risk_score'] < 0.6:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        return StrategyDecision(
            primary_action=best_option['action'],
            confidence=best_option['confidence'],
            expected_position=race_state.position + best_option['position_impact'],
            expected_time_delta=best_option['time_impact'],
            risk_level=risk_level,
            reasoning=reasoning,
            alternative_actions=alternatives,
            should_execute_now=should_execute
        )
    
    def _apply_feedback_adjustments(self, evaluation: Dict, 
                                   action: StrategyAction) -> Dict:
        """
        Adjust confidence based on historical performance of this action
        
        Feedback loop: Learn from past decisions
        """
        if action not in self.action_outcomes or len(self.action_outcomes[action]) == 0:
            return evaluation  # No history yet
        
        # Calculate historical success rate
        outcomes = self.action_outcomes[action]
        success_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
        
        # Adjust confidence
        adjustment_factor = 0.8 + (success_rate * 0.4)  # 0.8 to 1.2 range
        evaluation['confidence'] *= adjustment_factor
        evaluation['confidence'] = min(0.95, max(0.2, evaluation['confidence']))
        
        return evaluation
    
    def _update_feedback_loop(self, decision: StrategyDecision, 
                             race_state: RaceState):
        """Update feedback data for future decisions"""
        self.decision_history.append(decision)
        
        # Track confidence trends
        action_key = decision.primary_action.value
        if action_key not in self.confidence_trends:
            self.confidence_trends[action_key] = []
        self.confidence_trends[action_key].append(decision.confidence)
    
    def update_action_outcome(self, action: StrategyAction, outcome_value: float):
        """
        Called after action execution to record result
        
        outcome_value: positive = good, negative = bad
        """
        self.action_outcomes[action].append(outcome_value)
        
        # Keep only recent history
        if len(self.action_outcomes[action]) > 50:
            self.action_outcomes[action] = self.action_outcomes[action][-50:]
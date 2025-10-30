"""
Machine Learning Strategy Layer
Train on synthetic race data to predict optimal actions
"""

import numpy as np
from typing import Dict, List, Tuple
import pickle
from collections import deque

class RaceSimulator:
    """Generate synthetic race data for training"""
    
    def __init__(self):
        self.lap_time_base = 90.0  # seconds
        
    def simulate_race(self, total_laps: int = 50, 
                     strategy_sequence: List[str] = None) -> Tuple[List[Dict], float]:
        """
        Simulate a full race with given strategy
        
        Returns: (lap_data, final_position)
        """
        race_data = []
        
        # Initial state
        position = np.random.randint(3, 8)
        tyre_wear = 0
        tyre_age = 0
        fuel = 110
        gap_ahead = np.random.uniform(1.0, 5.0)
        gap_behind = np.random.uniform(1.0, 5.0)
        
        for lap in range(1, total_laps + 1):
            # Execute strategy if provided
            action_taken = None
            if strategy_sequence and len(strategy_sequence) >= lap:
                action_taken = strategy_sequence[lap - 1]
            
            # Simulate lap
            lap_time = self._simulate_lap_time(
                tyre_wear, tyre_age, fuel, position, action_taken
            )
            
            # Update state
            tyre_wear += np.random.uniform(2.0, 3.5)
            tyre_age += 1
            fuel -= 1.8
            
            # Handle pit stop
            if action_taken == 'PIT':
                lap_time += 22.0  # Pit stop time
                tyre_wear = 0
                tyre_age = 0
                position += np.random.choice([0, 1, 2])  # May lose positions
            
            # Handle overtake
            if action_taken == 'OVERTAKE':
                success_prob = self._calculate_overtake_success(
                    tyre_wear, gap_ahead
                )
                if np.random.random() < success_prob:
                    position = max(1, position - 1)
                    gap_ahead = np.random.uniform(0.5, 2.0)
                else:
                    lap_time += 0.5  # Failed overtake costs time
            
            # Natural position changes
            position_noise = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
            position = max(1, min(20, position + position_noise))
            
            # Update gaps
            gap_ahead += np.random.uniform(-0.3, 0.3)
            gap_behind += np.random.uniform(-0.3, 0.3)
            gap_ahead = max(0.1, gap_ahead)
            gap_behind = max(0.1, gap_behind)
            
            # Tyre cliff effect
            if tyre_wear > 80:
                lap_time += (tyre_wear - 80) * 0.1
            
            race_data.append({
                'lap': lap,
                'position': position,
                'tyre_wear': min(100, tyre_wear),
                'tyre_age': tyre_age,
                'fuel': max(0, fuel),
                'gap_ahead': gap_ahead,
                'gap_behind': gap_behind,
                'lap_time': lap_time,
                'action': action_taken or 'HOLD'
            })
        
        final_position = position
        return race_data, final_position
    
    def _simulate_lap_time(self, wear: float, age: int, fuel: float,
                          position: int, action: str) -> float:
        """Calculate lap time based on conditions"""
        base_time = self.lap_time_base
        
        # Wear degradation
        wear_penalty = (wear / 100) * 2.0
        
        # Fuel effect (lighter = faster)
        fuel_benefit = (110 - fuel) * 0.01
        
        # Position effect (traffic)
        traffic_penalty = max(0, (10 - position) * 0.05)
        
        # Pace mode
        pace_adjustment = 0
        if action == 'PUSH':
            pace_adjustment = -0.4
        elif action == 'CONSERVE':
            pace_adjustment = 0.3
        
        # Random variation
        noise = np.random.normal(0, 0.2)
        
        lap_time = base_time + wear_penalty - fuel_benefit + traffic_penalty + pace_adjustment + noise
        return max(85.0, lap_time)
    
    def _calculate_overtake_success(self, our_wear: float, gap: float) -> float:
        """Calculate overtake success probability"""
        base_prob = 0.3
        
        # Better tyres = higher chance
        if our_wear < 30:
            base_prob += 0.2
        
        # Smaller gap = easier
        if gap < 1.0:
            base_prob += 0.3
        elif gap < 2.0:
            base_prob += 0.15
        
        return min(0.9, base_prob)


class StrategyPolicyNetwork:
    """
    Simple neural network policy for strategy decisions
    
    Input: [tyre_wear, fuel, gap_ahead, gap_behind, position, lap, tyre_age]
    Output: [prob_overtake, prob_pit, prob_hold, prob_push, prob_conserve]
    """
    
    def __init__(self, input_size=7, hidden_size=32, output_size=5):
        # Initialize weights (simple 2-layer network)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
        
        self.learning_rate = 0.01
        self.action_names = ['OVERTAKE', 'PIT', 'HOLD', 'PUSH', 'CONSERVE']
    
    def forward(self, state: np.ndarray, return_hidden: bool = False) -> np.ndarray:
        """Forward pass through network"""
        # Hidden layer with ReLU
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Output layer with softmax
        z2 = np.dot(a1, self.W2) + self.b2
        exp_z2 = np.exp(z2 - np.max(z2))  # Numerical stability
        probabilities = exp_z2 / np.sum(exp_z2)
        
        if return_hidden:
            return probabilities, a1
        return probabilities
    
    def predict_action(self, race_state: Dict) -> Tuple[str, float]:
        """Predict best action for current state"""
        state_vector = self._state_to_vector(race_state)
        probabilities = self.forward(state_vector)
        
        best_action_idx = np.argmax(probabilities)
        best_action = self.action_names[best_action_idx]
        confidence = probabilities[best_action_idx]
        
        return best_action, float(confidence)
    

    def train_episode(self, race_data: List[Dict], final_position: float):
        """
        Train network on race outcome
        
        Uses REINFORCE algorithm (policy gradient)
        """
        states = []
        actions = []
        
        # Extract state-action pairs
        for lap_data in race_data:
            state = self._state_to_vector(lap_data)
            action = lap_data.get('action', 'HOLD')
            
            states.append(state)
            if action in self.action_names:
                actions.append(self.action_names.index(action))
            else:
                actions.append(2)  # Default to HOLD
        
        # Calculate reward (better position = higher reward)
        reward = (20 - final_position) / 20.0  # Normalize 0-1
        
        # Update weights using policy gradient
        for state, action_idx in zip(states, actions):
            # Forward pass with hidden activation
            probs, a1 = self.forward(state, return_hidden=True)
            
            # Compute gradient of log-prob w.r.t. logits: one_hot - probs
            grad_z2 = np.zeros_like(probs)
            grad_z2[action_idx] = 1.0
            grad_z2 -= probs  # shape: (5,)
            
            # Correct gradient for W2: outer product of hidden activation and grad_z2
            grad_W2 = np.outer(a1, grad_z2)  # shape: (32, 5)
            
            # Update W2
            self.W2 += self.learning_rate * reward * grad_W2
    
    def _state_to_vector(self, race_state: Dict) -> np.ndarray:
        """Convert race state dict to input vector"""
        return np.array([
            race_state.get('tyre_wear', 0) / 100,  # Normalize 0-1
            race_state.get('fuel', 50) / 110,
            min(race_state.get('gap_ahead', 5), 10) / 10,
            min(race_state.get('gap_behind', 5), 10) / 10,
            race_state.get('position', 10) / 20,
            race_state.get('lap', 25) / 50,
            race_state.get('tyre_age', 0) / 30
        ])
    
    def save(self, filepath: str):
        """Save model weights"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2
            }, f)
    
    def load(self, filepath: str):
        """Load model weights"""
        try:
            with open(filepath, 'rb') as f:
                weights = pickle.load(f)
                self.W1 = weights['W1']
                self.b1 = weights['b1']
                self.W2 = weights['W2']
                self.b2 = weights['b2']
            return True
        except:
            return False


class MLStrategyTrainer:
    """Train ML model on synthetic race data"""
    
    def __init__(self):
        self.simulator = RaceSimulator()
        self.policy = StrategyPolicyNetwork()
        self.training_history = []
    
    def train(self, num_races: int = 1000):
        """
        Train policy network on synthetic races
        
        Explores different strategies and learns from outcomes
        """
        print(f"Training ML strategy model on {num_races} synthetic races...")
        
        for race_num in range(num_races):
            # Generate random strategy sequence
            strategy = self._generate_random_strategy(50)
            
            # Simulate race
            race_data, final_position = self.simulator.simulate_race(
                total_laps=50,
                strategy_sequence=strategy
            )
            
            # Train on this episode
            self.policy.train_episode(race_data, final_position)
            
            # Track performance
            self.training_history.append({
                'race': race_num,
                'final_position': final_position,
                'strategy': strategy
            })
            
            if (race_num + 1) % 100 == 0:
                avg_position = np.mean([h['final_position'] for h in self.training_history[-100:]])
                print(f"  Race {race_num + 1}/{num_races} - Avg position: {avg_position:.2f}")
        
        print("✅ Training complete!")
        
        # Save model
        self.policy.save('strategy_policy.pkl')
        print("💾 Model saved to strategy_policy.pkl")
    
    def _generate_random_strategy(self, total_laps: int) -> List[str]:
        """Generate random but semi-reasonable strategy"""
        strategy = ['HOLD'] * total_laps
        
        # Add 1-2 pit stops
        num_pits = np.random.choice([1, 2], p=[0.7, 0.3])
        for _ in range(num_pits):
            pit_lap = np.random.randint(10, total_laps - 10)
            strategy[pit_lap] = 'PIT'
        
        # Add some overtake attempts
        num_overtakes = np.random.randint(0, 5)
        for _ in range(num_overtakes):
            overtake_lap = np.random.randint(5, total_laps - 5)
            if strategy[overtake_lap] != 'PIT':
                strategy[overtake_lap] = 'OVERTAKE'
        
        # Add pace variations
        for lap in range(total_laps):
            if strategy[lap] == 'HOLD' and np.random.random() < 0.2:
                strategy[lap] = np.random.choice(['PUSH', 'CONSERVE'])
        
        return strategy
    
    def evaluate(self, num_test_races: int = 100):
        """Evaluate trained policy"""
        print(f"\n🧪 Evaluating policy on {num_test_races} test races...")
        
        positions = []
        
        for _ in range(num_test_races):
            # Simulate race using policy decisions
            race_data = []
            state = {'lap': 1, 'position': 5, 'tyre_wear': 0, 'fuel': 110, 
                    'gap_ahead': 2.0, 'gap_behind': 2.0, 'tyre_age': 0}
            
            strategy_sequence = []
            for lap in range(1, 51):
                state['lap'] = lap
                
                # Predict action
                action, conf = self.policy.predict_action(state)
                strategy_sequence.append(action)
                
                # Update state (simplified)
                state['tyre_wear'] = min(100, state['tyre_wear'] + 2.5)
                state['fuel'] = max(0, state['fuel'] - 1.8)
                state['tyre_age'] += 1
                
                if action == 'PIT':
                    state['tyre_wear'] = 0
                    state['tyre_age'] = 0
            
            # Simulate full race
            _, final_pos = self.simulator.simulate_race(50, strategy_sequence)
            positions.append(final_pos)
        
        avg_pos = np.mean(positions)
        print(f"📊 Average finishing position: {avg_pos:.2f}")
        print(f"🏆 Podium finishes: {sum(1 for p in positions if p <= 3)}/{num_test_races}")


# Quick training script
if __name__ == '__main__':
    trainer = MLStrategyTrainer()
    
    # Train
    trainer.train(num_races=1000)
    
    # Evaluate
    trainer.evaluate(num_test_races=100)
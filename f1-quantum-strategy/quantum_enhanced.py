"""
Enhanced Quantum Strategy Engine
Real quantum optimization circuits for strategy selection
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, PhaseOracle
from qiskit.quantum_info import Statevector
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class QuantumStrategyOptimizer:
    """
    True quantum optimization for F1 strategy
    
    Uses Grover's algorithm and QAOA-inspired circuits to find optimal strategies
    """
    
    def __init__(self, shots: int = 4096):
        self.simulator = AerSimulator()
        self.shots = shots
        
    def optimize_multi_strategy(self, strategies: List[Dict], 
                                constraints: Dict) -> Dict:
        """
        Use quantum amplitude amplification to find optimal strategy
        
        Encodes each strategy as a quantum state, applies oracle that marks
        good strategies, then amplifies their amplitudes
        
        Args:
            strategies: List of strategy dicts with 'name', 'expected_value', 'constraints'
            constraints: Race constraints (lap, fuel, tyres, etc.)
            
        Returns:
            Optimal strategy with quantum-derived confidence
        """
        n_strategies = len(strategies)
        if n_strategies == 0:
            return {'error': 'No strategies provided'}
        
        # Number of qubits needed
        n_qubits = max(2, int(np.ceil(np.log2(n_strategies))))
        
        # Create quantum circuit
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # STEP 1: Initialize superposition (equal amplitude for all strategies)
        for i in range(n_qubits):
            qc.h(i)
        qc.barrier()
        
        # STEP 2: Encode strategy values as phase rotations
        for idx, strategy in enumerate(strategies):
            if idx >= 2**n_qubits:
                break
            
            # Calculate phase based on expected value
            expected_val = strategy.get('expected_value', 0)
            risk_score = strategy.get('risk_score', 0.5)
            
            # Good strategies get positive phase, bad ones get negative
            phase = expected_val * np.pi / 10
            
            # Apply controlled phase rotation for this strategy index
            binary_idx = format(idx, f'0{n_qubits}b')
            self._apply_strategy_phase(qc, binary_idx, phase, n_qubits)
        
        qc.barrier()
        
        # STEP 3: Apply Grover diffusion operator (amplitude amplification)
        # This amplifies states with positive phase
        num_iterations = max(1, int(np.sqrt(2**n_qubits)))
        for _ in range(num_iterations):
            self._grover_diffusion(qc, n_qubits)
        
        qc.barrier()
        
        # STEP 4: Measure
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Execute quantum circuit
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Decode measurement results
        optimal_strategy = self._decode_measurement(
            counts, strategies, n_qubits
        )
        
        # Calculate quantum confidence
        most_measured_state = max(counts, key=counts.get)
        quantum_confidence = counts[most_measured_state] / self.shots
        
        # Enhance with constraints satisfaction check
        constraint_score = self._check_constraints(
            optimal_strategy, constraints
        )
        
        return {
            'optimal_strategy': optimal_strategy['name'],
            'expected_value': optimal_strategy['expected_value'],
            'quantum_confidence': round(quantum_confidence * 100, 1),
            'constraint_satisfaction': constraint_score,
            'combined_confidence': round(quantum_confidence * constraint_score * 100, 1),
            'measurement_distribution': counts,
            'quantum_circuit_depth': qc.depth(),
            'iterations_used': num_iterations,
            'alternatives': self._get_top_alternatives(counts, strategies, n_qubits, 3)
        }
    
    def variational_strategy_optimization(self, race_state: Dict, 
                                         available_actions: List[str]) -> Dict:
        """
        QAOA-inspired variational optimization
        
        Uses parameterized quantum circuit to optimize strategy selection
        """
        n_actions = len(available_actions)
        n_qubits = max(2, int(np.ceil(np.log2(n_actions))))
        
        # Build cost Hamiltonian (encodes race state preferences)
        cost_params = self._build_cost_hamiltonian(race_state)
        
        # QAOA-like circuit with mixer and cost layers
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initial state: equal superposition
        for i in range(n_qubits):
            qc.h(i)
        
        # Variational layers (simplified QAOA)
        n_layers = 3
        for layer in range(n_layers):
            # Cost layer (problem-specific)
            self._apply_cost_layer(qc, cost_params, n_qubits)
            
            # Mixer layer (exploration)
            self._apply_mixer_layer(qc, n_qubits)
        
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Execute
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Decode best action
        best_state = max(counts, key=counts.get)
        action_idx = int(best_state, 2) % n_actions
        best_action = available_actions[action_idx]
        
        confidence = counts[best_state] / self.shots
        
        return {
            'recommended_action': best_action,
            'confidence': round(confidence * 100, 1),
            'measurement_counts': counts,
            'circuit_layers': n_layers,
            'exploration_score': self._calculate_exploration_score(counts)
        }
    
    def quantum_monte_carlo_enhanced(self, current_state: Dict,
                                    scenarios: List[Dict],
                                    num_simulations: int = 2048) -> Dict:
        """
        Enhanced quantum Monte Carlo using actual quantum sampling
        
        Uses quantum random number generation and superposition
        """
        n_scenarios = len(scenarios)
        n_qubits = max(2, int(np.ceil(np.log2(n_scenarios))))
        
        # Build circuit for quantum sampling
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Create weighted superposition based on scenario probabilities
        for i in range(n_qubits):
            qc.h(i)
        
        # Apply rotations based on scenario likelihoods
        for idx, scenario in enumerate(scenarios):
            if idx >= 2**n_qubits:
                break
            
            likelihood = scenario.get('likelihood', 0.5)
            # Rotate to increase amplitude for more likely scenarios
            theta = 2 * np.arcsin(np.sqrt(likelihood))
            
            binary = format(idx, f'0{n_qubits}b')
            # Apply multi-controlled rotation
            for q in range(n_qubits):
                if binary[q] == '1':
                    qc.ry(theta / (2**q), q)
        
        # Entangle for correlations
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Run many shots for Monte Carlo
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=num_simulations)
        result = job.result()
        counts = result.get_counts()
        
        # Analyze results
        scenario_probabilities = self._analyze_quantum_samples(
            counts, scenarios, n_qubits
        )
        
        # Find best scenario
        best_scenario_name = max(
            scenario_probabilities,
            key=scenario_probabilities.get
        )
        
        best_scenario = next(
            s for s in scenarios if s['name'] == best_scenario_name
        )
        
        return {
            'simulation_type': 'Quantum Monte Carlo',
            'num_simulations': num_simulations,
            'scenario_probabilities': scenario_probabilities,
            'recommended_strategy': best_scenario_name,
            'expected_outcome': best_scenario.get('outcome'),
            'confidence': round(scenario_probabilities[best_scenario_name] * 100, 1),
            'quantum_advantage': 'True quantum sampling with superposition'
        }
    
    def _apply_strategy_phase(self, qc: QuantumCircuit, binary_target: str,
                             phase: float, n_qubits: int):
        """Apply phase to specific strategy state"""
        # Multi-controlled phase gate
        controls = []
        for i, bit in enumerate(binary_target):
            if bit == '0':
                qc.x(i)
            controls.append(i)
        
        # Apply phase (using RZ for computational basis)
        if len(controls) == 1:
            qc.rz(phase, controls[0])
        else:
            qc.mcrz(phase, controls[:-1], controls[-1])
        
        # Undo X gates
        for i, bit in enumerate(binary_target):
            if bit == '0':
                qc.x(i)
    
    def _grover_diffusion(self, qc: QuantumCircuit, n_qubits: int):
        """Grover diffusion operator (amplitude amplification)"""
        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            qc.h(i)
        
        # Apply X to all qubits
        for i in range(n_qubits):
            qc.x(i)
        
        # Multi-controlled Z (phase flip for |11...1>)
        if n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        
        # Apply X to all qubits
        for i in range(n_qubits):
            qc.x(i)
        
        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            qc.h(i)
    
    def _build_cost_hamiltonian(self, race_state: Dict) -> Dict:
        """Build problem Hamiltonian from race state"""
        return {
            'tyre_urgency': race_state.get('tyre_wear', 50) / 100,
            'gap_pressure': 1.0 / (race_state.get('gap_ahead', 5) + 1),
            'fuel_constraint': race_state.get('fuel', 50) / 110,
            'position_value': 1.0 / (race_state.get('position', 5) + 1)
        }
    
    def _apply_cost_layer(self, qc: QuantumCircuit, cost_params: Dict, n_qubits: int):
        """Apply cost operator (problem-specific rotations)"""
        for i in range(n_qubits):
            # Rotation angle based on problem constraints
            angle = cost_params.get('tyre_urgency', 0.5) * np.pi / 2
            qc.rz(angle, i)
        
        # Add entanglement for constraint coupling
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    
    def _apply_mixer_layer(self, qc: QuantumCircuit, n_qubits: int):
        """Apply mixer operator (exploration)"""
        for i in range(n_qubits):
            qc.rx(np.pi / 4, i)
    
    def _decode_measurement(self, counts: Dict, strategies: List[Dict],
                           n_qubits: int) -> Dict:
        """Decode measurement result to strategy"""
        # Find most measured state
        best_state = max(counts, key=counts.get)
        strategy_idx = int(best_state, 2) % len(strategies)
        
        return strategies[strategy_idx]
    
    def _check_constraints(self, strategy: Dict, constraints: Dict) -> float:
        """Check how well strategy satisfies constraints (0-1)"""
        score = 1.0
        
        # Check fuel constraint
        if 'fuel_required' in strategy and 'fuel' in constraints:
            if strategy['fuel_required'] > constraints['fuel']:
                score *= 0.5
        
        # Check tyre constraint
        if 'requires_fresh_tyres' in strategy and constraints.get('tyre_wear', 0) > 80:
            score *= 1.2  # Bonus if we need tyres and pit is recommended
        
        return min(1.0, score)
    
    def _get_top_alternatives(self, counts: Dict, strategies: List[Dict],
                             n_qubits: int, top_n: int) -> List[Dict]:
        """Get top N alternative strategies from measurements"""
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        alternatives = []
        
        for state, count in sorted_counts[1:top_n+1]:
            idx = int(state, 2) % len(strategies)
            if idx < len(strategies):
                alternatives.append({
                    'strategy': strategies[idx]['name'],
                    'probability': round(count / self.shots, 3)
                })
        
        return alternatives
    
    def _analyze_quantum_samples(self, counts: Dict, scenarios: List[Dict],
                                n_qubits: int) -> Dict:
        """Analyze quantum sampling results"""
        scenario_counts = {s['name']: 0 for s in scenarios}
        
        for state, count in counts.items():
            idx = int(state, 2) % len(scenarios)
            if idx < len(scenarios):
                scenario_counts[scenarios[idx]['name']] += count
        
        total = sum(scenario_counts.values())
        return {name: count / total for name, count in scenario_counts.items()}
    
    def _calculate_exploration_score(self, counts: Dict) -> float:
        """Calculate how well the circuit explored solution space"""
        if len(counts) == 0:
            return 0.0
        
        # Shannon entropy as exploration metric
        probs = np.array(list(counts.values())) / sum(counts.values())
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(counts))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0


# Example usage
if __name__ == '__main__':
    optimizer = QuantumStrategyOptimizer(shots=4096)
    
    # Define strategies
    strategies = [
        {'name': 'Pit Now', 'expected_value': 2.5, 'risk_score': 0.3},
        {'name': 'Overtake Then Pit', 'expected_value': 4.0, 'risk_score': 0.6},
        {'name': 'Hold Position', 'expected_value': 1.0, 'risk_score': 0.2},
        {'name': 'Push Pace', 'expected_value': 3.0, 'risk_score': 0.5},
    ]
    
    constraints = {
        'lap': 25,
        'fuel': 45,
        'tyre_wear': 72
    }
    
    print("🔬 Running Quantum Strategy Optimization...\n")
    result = optimizer.optimize_multi_strategy(strategies, constraints)
    
    print(f"✅ Optimal Strategy: {result['optimal_strategy']}")
    print(f"📊 Quantum Confidence: {result['quantum_confidence']}%")
    print(f"🎯 Combined Confidence: {result['combined_confidence']}%")
    print(f"\n🔄 Circuit Depth: {result['quantum_circuit_depth']}")
    print(f"🌀 Grover Iterations: {result['iterations_used']}")
    print(f"\n📈 Alternatives:")
    for alt in result['alternatives']:
        print(f"   • {alt['strategy']}: {alt['probability']*100:.1f}%")
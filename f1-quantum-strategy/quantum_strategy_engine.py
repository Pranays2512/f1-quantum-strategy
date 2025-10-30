"""
Quantum Strategy Engine
Uses Qiskit to perform quantum optimization for F1 race strategies
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
import numpy as np
from typing import Dict, List

class QuantumStrategyEngine:
    def __init__(self):
        self.simulator = AerSimulator()
        self.shots = 1024  # Number of quantum measurements
        
    def optimize_pit_strategy(self, current_lap: int, tyre_wear: float, 
                             tyre_temps: Dict, total_laps: int,
                             competitors: List, track_conditions) -> Dict:
        """
        Use quantum computing to evaluate multiple pit stop strategies simultaneously
        Returns optimal pit stop timing and tyre compound choice
        """
        
        # Calculate urgency factors
        temp_urgency = self._calculate_temp_urgency(tyre_temps)
        wear_urgency = tyre_wear / 100.0
    
        # NEW: Combined urgency with exponential scaling
        combined_urgency = max(temp_urgency, wear_urgency)
        if tyre_wear > 80 and temp_urgency > 0.7:
            combined_urgency = min(0.98, combined_urgency * 1.3)  # Extra urgent
        
        # NEW: Calculate laps since pit (if we can estimate)
        estimated_stint_length = max(1, int(tyre_wear / 3))  # Rough estimate
        
        # CRITICAL CONDITION CHECK (bypass quantum if urgent)
        if combined_urgency > 0.9 or (tyre_wear > 85 and temp_urgency > 0.75):
            return {
                "recommendation": "URGENT - PIT NOW!",
                "optimal_lap": current_lap + 1,
                "laps_until_pit": 1,
                "tyre_compound": "Hard" if track_conditions.rainfall < 20 else "Intermediate",
                "confidence": 95.0,
                "expected_time_impact": -25.0,
                "reasoning": f"CRITICAL: Wear={tyre_wear}%, Temp={max(tyre_temps.values())}°C | Emergency pit required",
                "alternative_strategies": []
            }
        
        # Define possible pit windows (next 10 laps)
        possible_pit_laps = []
        for i in range(1, min(11, total_laps - current_lap + 1)):
            possible_pit_laps.append(current_lap + i)
            
        # Define possible pit windows (next 10 laps)
        
        if not possible_pit_laps:
            return {
                "recommendation": "No pit stop needed",
                "optimal_lap": None,
                "tyre_compound": "current",
                "confidence": 0,
                "reasoning": "Race ending soon"
            }
        
        # Create quantum circuit to evaluate strategies
        n_qubits = min(4, int(np.ceil(np.log2(len(possible_pit_laps)))) + 2)
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize superposition - explore all pit timing options simultaneously
        for i in range(n_qubits):
            qc.h(i)  # Hadamard gate creates superposition
        
        # Encode strategy parameters using quantum gates
        # Qubit 0-1: Pit timing (early/mid/late window)
        # Qubit 2: Tyre compound (soft=0, hard=1)
        # Qubit 3: Urgency factor
        
        # Apply urgency-based rotations
        urgency = max(temp_urgency, wear_urgency)
        qc.ry(urgency * np.pi, 0)  # Rotate based on urgency
        
        # Temperature influence
        if temp_urgency > 0.7:  # Hot tyres
            qc.x(2)  # Flip to prefer harder compound
        
        # Entangle timing with tyre choice
        qc.cx(0, 2)  # Create quantum correlation
        qc.cx(1, 3)
        
        # Apply interference to amplify good strategies
        qc.barrier()
        for i in range(n_qubits - 1):
            qc.cz(i, i + 1)  # Controlled-Z gates for interference
        
        # Measure all qubits
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Execute quantum circuit
        job = self.simulator.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Interpret quantum results
        best_strategy = max(counts, key=counts.get)
        confidence = counts[best_strategy] / self.shots
        
        # Decode quantum result to pit strategy
        strategy_bits = [int(b) for b in best_strategy]
        
        # Determine optimal pit lap
        timing_index = strategy_bits[0] + (strategy_bits[1] << 1)
        optimal_lap_index = min(timing_index, len(possible_pit_laps) - 1)
        optimal_lap = possible_pit_laps[optimal_lap_index]
        
        # Determine tyre compound
        tyre_compound = "Hard" if strategy_bits[2] == 1 else "Soft"
        
        # Adjust based on weather
        if track_conditions.rainfall > 50:
            tyre_compound = "Intermediate"
        elif track_conditions.rainfall > 20:
            tyre_compound = "Soft"
        
        # Calculate expected time impact
        laps_on_old_tyres = optimal_lap - current_lap
        time_loss_pitstop = 22.0  # seconds
        time_gain_per_lap = 0.3 * (tyre_wear / 100)  # gain from fresh tyres
        laps_after_pit = total_laps - optimal_lap
        
        net_time_impact = -time_loss_pitstop + (time_gain_per_lap * laps_after_pit)
        
        recommendation = "URGENT - Pit now!" if urgency > 0.85 else f"Pit on lap {optimal_lap}"
        
        return {
            "recommendation": recommendation,
            "optimal_lap": optimal_lap,
            "laps_until_pit": optimal_lap - current_lap,
            "tyre_compound": tyre_compound,
            "confidence": round(confidence * 100, 1),
            "expected_time_impact": round(net_time_impact, 2),
            "reasoning": self._generate_pit_reasoning(urgency, temp_urgency, wear_urgency, tyre_compound),
            "alternative_strategies": self._get_alternatives(counts, possible_pit_laps)
        }
    
    def optimize_pace_strategy(self, current_position: int, fuel_load: float,
                               tyre_condition: float, laps_remaining: int) -> Dict:
        """
        Quantum optimization for pace management
        Balances speed vs tyre/fuel conservation
        """
        
        # Create quantum circuit for pace evaluation
        qc = QuantumCircuit(3, 3)
        
        # Superposition of pace strategies
        qc.h(0)  # Qubit 0: Push/Conserve
        qc.h(1)  # Qubit 1: Aggressive/Moderate
        qc.h(2)  # Qubit 2: Fuel saving mode
        
        # Encode current situation
        tyre_factor = 1 - (tyre_condition / 100)
        fuel_factor = fuel_load / 110  # Assuming max ~110kg
        position_pressure = 1 / (current_position + 1)  # Higher pressure if leading
        
        # Apply rotations based on situation
        qc.ry(tyre_factor * np.pi / 2, 0)
        qc.ry(position_pressure * np.pi / 2, 1)
        qc.ry(fuel_factor * np.pi / 4, 2)
        
        # Entangle pace decisions
        qc.cx(0, 1)
        qc.cx(1, 2)
        
        # Measure
        qc.measure(range(3), range(3))
        
        # Execute
        job = self.simulator.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Interpret results
        best_pace = max(counts, key=counts.get)
        pace_bits = [int(b) for b in best_pace]
        
        # Decode pace strategy
        if pace_bits[0] == 1 and pace_bits[1] == 1:
            pace_mode = "ATTACK"
            lap_time_target = "Push -0.5s per lap"
        elif pace_bits[0] == 1:
            pace_mode = "PUSH"
            lap_time_target = "Push -0.3s per lap"
        elif pace_bits[1] == 0:
            pace_mode = "CONSERVE"
            lap_time_target = "Maintain +0.2s per lap"
        else:
            pace_mode = "BALANCED"
            lap_time_target = "Current pace"
        
        fuel_save = "Enable fuel saving mode" if pace_bits[2] == 1 else "Normal fuel mode"
        
        return {
            "pace_mode": pace_mode,
            "lap_time_target": lap_time_target,
            "fuel_strategy": fuel_save,
            "tyre_management": "Conservative" if tyre_condition > 60 else "Can push",
            "laps_remaining": laps_remaining,
            "recommendation": f"{pace_mode}: {lap_time_target}"
        }
    
    def _calculate_temp_urgency(self, tyre_temps: Dict) -> float:
        """Calculate urgency based on tyre temperatures"""
        temps = [tyre_temps.get(pos, 0) for pos in ['FL', 'FR', 'RL', 'RR']]
        avg_temp = sum(temps) / len(temps)
        
        # Optimal range: 90-105°C
        if avg_temp > 110:
            return 0.9  # Very hot - urgent
        elif avg_temp > 105:
            return 0.6  # Hot
        elif avg_temp < 80:
            return 0.5  # Too cold
        else:
            return 0.2  # Good range
    
    def _generate_pit_reasoning(self, urgency: float, temp_urgency: float, 
                                wear_urgency: float, compound: str) -> str:
        """Generate human-readable reasoning for pit recommendation"""
        reasons = []
        
        if wear_urgency > 0.7:
            reasons.append("High tyre wear detected")
        if temp_urgency > 0.7:
            reasons.append("Tyre temperatures critical")
        if urgency < 0.3:
            reasons.append("Tyres still in good condition")
        
        reasons.append(f"{compound} compound recommended")
        
        return " | ".join(reasons)
    
    def _get_alternatives(self, counts: Dict, possible_laps: List) -> List[Dict]:
        """Get alternative strategies from quantum results"""
        sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        alternatives = []
        
        for i, (state, count) in enumerate(sorted_results[1:4]):  # Top 3 alternatives
            timing_index = int(state[0]) + (int(state[1]) << 1)
            lap_index = min(timing_index, len(possible_laps) - 1)
            
            alternatives.append({
                "lap": possible_laps[lap_index],
                "confidence": round((count / self.shots) * 100, 1),
                "compound": "Hard" if int(state[2]) == 1 else "Soft"
            })
        
        return alternatives
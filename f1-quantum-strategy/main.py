"""
F1 Quantum Strategy Optimization Backend
Main application file with FastAPI endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime
from collections import deque

# Import our quantum strategy engine
from quantum_strategy_engine import QuantumStrategyEngine
from strategy_analyzer import StrategyAnalyzer
from feature_engineering import extract_features_simple

# Global buffer for recent telemetry (simple in-memory)
telemetry_history = {}  # car_id -> deque of samples
MAX_HISTORY = 50

app = FastAPI(title="F1 Quantum Strategy API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
quantum_engine = QuantumStrategyEngine()
strategy_analyzer = StrategyAnalyzer()

# ==================== DATA MODELS ====================

class TyreData(BaseModel):
    FL: float  # Front Left temperature
    FR: float  # Front Right temperature
    RL: float  # Rear Left temperature
    RR: float  # Rear Right temperature

class OurCarData(BaseModel):
    position: int
    speed: float
    tyre_temp: TyreData
    tyre_wear: float  # 0-100%
    fuel_load: float  # kg
    lap_time: float
    current_lap: int
    sector_times: List[float]
    slow_sectors: List[str]  # Where our car is slow

class CompetitorData(BaseModel):
    car_id: str
    position: int
    speed: float
    gap: float  # seconds ahead/behind
    slow_zones: List[str]  # Where they are slow
    tyre_age: int  # laps on current tyres

class TrackConditions(BaseModel):
    temperature: float
    rainfall: float  # 0-100%
    track_evolution: float  # grip level 0-100%

class RaceData(BaseModel):
    timestamp: int
    our_car: OurCarData
    competitors: List[CompetitorData]
    track_conditions: TrackConditions
    total_laps: int
    drs_zones: List[str]

class StrategyRecommendation(BaseModel):
    timestamp: str
    pit_stop_recommendation: Dict
    overtaking_opportunities: List[Dict]
    pace_strategy: Dict
    sector_optimization: Dict
    risk_assessment: str
    expected_time_gain: float

# ==================== HELPER FUNCTIONS ====================

def get_enhanced_recommendation(race_data: RaceData) -> dict:
    """Enhanced recommendation using feature engineering"""
    
    car_id = f"car_{race_data.our_car.position}"
    
    # Store in history
    if car_id not in telemetry_history:
        telemetry_history[car_id] = deque(maxlen=MAX_HISTORY)
    
    # Convert current data to sample
    sample = {
        'current_lap': race_data.our_car.current_lap,
        'tyre_wear': race_data.our_car.tyre_wear,
        'fuel_load': race_data.our_car.fuel_load,
        'tyre_temp': race_data.our_car.tyre_temp.dict(),
        'lap_time': race_data.our_car.lap_time,
        'position': race_data.our_car.position
    }
    telemetry_history[car_id].append(sample)
    
    # Extract features
    features = extract_features_simple(list(telemetry_history[car_id]))
    
    # Enhanced pit decision logic
    pit_rec = quantum_engine.optimize_pit_strategy(
        current_lap=race_data.our_car.current_lap,
        tyre_wear=race_data.our_car.tyre_wear,
        tyre_temps=race_data.our_car.tyre_temp.dict(),
        total_laps=race_data.total_laps,
        competitors=race_data.competitors,
        track_conditions=race_data.track_conditions
    )
    
    # ENHANCEMENT: Adjust confidence based on features
    if features['temp_trend'] > 5:  # Temps rising fast
        pit_rec['confidence'] = min(95, pit_rec['confidence'] + 15)
        pit_rec['reasoning'] += " | Rapid temp increase detected"
    
    if features['laps_on_stint'] > 20 and features['avg_tyre_temp'] > 105:
        pit_rec['recommendation'] = "URGENT - Pit NOW!"
        pit_rec['confidence'] = 95
    
    # Add feature explanation
    pit_rec['features_analyzed'] = {
        'stint_length': features['laps_on_stint'],
        'avg_temp': round(features['avg_tyre_temp'], 1),
        'temp_trend': round(features['temp_trend'], 1),
        'imbalance': round(features['temp_imbalance'], 1)
    }
    
    return pit_rec

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "F1 Quantum Strategy API",
        "status": "operational",
        "quantum_backend": "qiskit-aer simulator"
    }

@app.post("/api/strategy/analyze")
async def analyze_strategy(race_data: RaceData):
    """Enhanced with feature engineering"""
    try:
        # Use enhanced recommendation
        pit_recommendation = get_enhanced_recommendation(race_data)
        
        # Rest stays the same
        overtaking_opps = strategy_analyzer.find_overtaking_opportunities(
            our_car=race_data.our_car,
            competitors=race_data.competitors,
            drs_zones=race_data.drs_zones
        )
        
        pace_strategy = quantum_engine.optimize_pace_strategy(
            current_position=race_data.our_car.position,
            fuel_load=race_data.our_car.fuel_load,
            tyre_condition=race_data.our_car.tyre_wear,
            laps_remaining=race_data.total_laps - race_data.our_car.current_lap
        )
        
        sector_optimization = strategy_analyzer.optimize_sectors(
            our_slow_sectors=race_data.our_car.slow_sectors,
            competitor_slow_zones=[c.slow_zones for c in race_data.competitors],
            our_sector_times=race_data.our_car.sector_times
        )
        
        time_gain = strategy_analyzer.calculate_expected_gain(
            pit_recommendation,
            pace_strategy,
            sector_optimization
        )
        
        risk_level = strategy_analyzer.assess_risk(
            pit_recommendation,
            race_data.track_conditions,
            race_data.our_car.position
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'pit_stop_recommendation': pit_recommendation,
            'overtaking_opportunities': overtaking_opps,
            'pace_strategy': pace_strategy,
            'sector_optimization': sector_optimization,
            'risk_assessment': risk_level,
            'expected_time_gain': time_gain
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy analysis failed: {str(e)}")

@app.post("/api/strategy/pit-timing")
async def analyze_pit_timing(race_data: RaceData):
    """
    Dedicated endpoint for pit stop timing analysis
    """
    try:
        result = quantum_engine.optimize_pit_strategy(
            current_lap=race_data.our_car.current_lap,
            tyre_wear=race_data.our_car.tyre_wear,
            tyre_temps=race_data.our_car.tyre_temp.dict(),
            total_laps=race_data.total_laps,
            competitors=race_data.competitors,
            track_conditions=race_data.track_conditions
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategy/overtaking")
async def analyze_overtaking(race_data: RaceData):
    """
    Analyze best overtaking opportunities
    """
    try:
        opportunities = strategy_analyzer.find_overtaking_opportunities(
            our_car=race_data.our_car,
            competitors=race_data.competitors,
            drs_zones=race_data.drs_zones
        )
        return {"overtaking_opportunities": opportunities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategy/sector-analysis")
async def analyze_sectors(race_data: RaceData):
    """
    Detailed sector time analysis and optimization suggestions
    """
    try:
        analysis = strategy_analyzer.optimize_sectors(
            our_slow_sectors=race_data.our_car.slow_sectors,
            competitor_slow_zones=[c.slow_zones for c in race_data.competitors],
            our_sector_times=race_data.our_car.sector_times
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "quantum_simulator": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_stats():
    """Quick stats endpoint"""
    return {
        "cars_tracked": len(telemetry_history),
        "total_samples": sum(len(h) for h in telemetry_history.values()),
        "cars": {
            car_id: {
                "samples": len(history),
                "last_lap": history[-1].get('current_lap', 0) if history else 0
            }
            for car_id, history in telemetry_history.items()
        }
    }

if __name__ == "__main__":
    print("🏎️  Starting F1 Quantum Strategy Backend...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
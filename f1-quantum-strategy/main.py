"""
F1 Quantum Strategy Backend - WITH AUTONOMOUS EXECUTION
Real-time strategy analysis and automatic execution
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime
import logging
import asyncio
import json

# Import modules
from quantum_strategy_engine import QuantumStrategyEngine
from strategy_analyzer import StrategyAnalyzer
from pit_prediction import PitStopPredictor
from tyre_modeling import TyreModel
from weak_point_detector import WeakPointDetector
from autonomous_strategy_executor import AutonomousStrategyExecutor, TelemetryStreamHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Quantum Strategy API - Autonomous",
    version="4.0.0",
    description="AI-powered F1 race strategy with autonomous execution"
)

# CORS
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
pit_predictor = PitStopPredictor()
tyre_model = TyreModel()
weak_point_detector = WeakPointDetector()

# Initialize AUTONOMOUS EXECUTOR
autonomous_executor = AutonomousStrategyExecutor(ai_backend_url="http://localhost:8000")
telemetry_handler = TelemetryStreamHandler(autonomous_executor)

# Global state
telemetry_history = {}
performance_stats = {
    'total_requests': 0,
    'avg_response_time_ms': 0,
    'cars_tracked': 0,
    'total_samples': 0,
    'autonomous_executions': 0
}

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

# ==================== DATA MODELS ====================

class TyreData(BaseModel):
    FL: float
    FR: float
    RL: float
    RR: float

class OurCarData(BaseModel):
    position: int
    speed: float
    tyre_temp: TyreData
    tyre_wear: float
    fuel_load: float
    lap_time: float
    current_lap: int
    sector_times: List[float] = Field(default_factory=lambda: [30.0, 32.0, 28.0])
    slow_sectors: List[str] = Field(default_factory=list)

class CompetitorData(BaseModel):
    car_id: str
    position: int
    speed: float
    gap: float
    slow_zones: List[str] = Field(default_factory=list)
    tyre_age: int

class TrackConditions(BaseModel):
    temperature: float
    rainfall: float
    track_evolution: float

class RaceData(BaseModel):
    timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    our_car: OurCarData
    competitors: List[CompetitorData] = Field(default_factory=list)
    track_conditions: TrackConditions
    total_laps: int
    drs_zones: List[str] = Field(default_factory=list)

# ==================== HELPER FUNCTIONS ====================

def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def track_conditions_to_dict(track_conditions) -> Dict:
    """Convert TrackConditions object to dict safely"""
    if isinstance(track_conditions, dict):
        return track_conditions
    return {
        'temperature': getattr(track_conditions, 'temperature', 25),
        'rainfall': getattr(track_conditions, 'rainfall', 0),
        'track_evolution': getattr(track_conditions, 'track_evolution', 85)
    }

# ==================== WEBSOCKET FOR REAL-TIME UPDATES ====================

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"📡 WebSocket connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Receive telemetry from simulator
            data = await websocket.receive_text()
            telemetry = json.loads(data)
            
            # Update autonomous executor with live telemetry
            await telemetry_handler.handle_telemetry_update(telemetry)
            
            # Optionally send back acknowledgment
            await websocket.send_json({
                "status": "received",
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"📡 WebSocket disconnected. Remaining: {len(active_connections)}")

async def broadcast_execution(command: Dict):
    """Broadcast execution command to all connected clients"""
    for connection in active_connections:
        try:
            await connection.send_json({
                "type": "execution",
                "command": command,
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass

# ==================== MAIN ENDPOINTS ====================

@app.post("/api/strategy/analyze")
async def analyze_strategy_autonomous(race_data: RaceData):
    """
    Real-time strategy analysis WITH AUTONOMOUS EXECUTION
    
    This endpoint:
    1. Analyzes strategy using AI
    2. Automatically executes if confidence > 70%
    3. Broadcasts execution to connected clients
    """
    start_time = datetime.now()
    
    try:
        # Update stats
        performance_stats['total_requests'] += 1
        
        # Store telemetry
        car_id = f"car_{race_data.our_car.position}"
        if car_id not in telemetry_history:
            telemetry_history[car_id] = []
            performance_stats['cars_tracked'] += 1
        
        track_dict = track_conditions_to_dict(race_data.track_conditions)
        
        telemetry_sample = {
            'current_lap': race_data.our_car.current_lap,
            'tyre_wear': race_data.our_car.tyre_wear,
            'fuel_load': race_data.our_car.fuel_load,
            'tyre_temp': race_data.our_car.tyre_temp.model_dump(),
            'lap_time': race_data.our_car.lap_time,
            'position': race_data.our_car.position,
            'sector_times': race_data.our_car.sector_times[:3],
            'speed': race_data.our_car.speed
        }
        telemetry_history[car_id].append(telemetry_sample)
        performance_stats['total_samples'] += 1
        
        if len(telemetry_history[car_id]) > 50:
            telemetry_history[car_id] = telemetry_history[car_id][-50:]
        
        # 1. QUANTUM PIT STRATEGY
        pit_recommendation = quantum_engine.optimize_pit_strategy(
            race_data.our_car.current_lap,
            race_data.our_car.tyre_wear,
            race_data.our_car.tyre_temp.model_dump(),
            race_data.total_laps,
            race_data.competitors,
            track_dict
        )
        
        # 2. PACE STRATEGY
        pace_strategy = quantum_engine.optimize_pace_strategy(
            race_data.our_car.position,
            race_data.our_car.fuel_load,
            race_data.our_car.tyre_wear,
            race_data.total_laps - race_data.our_car.current_lap
        )
        
        # 3. OVERTAKING OPPORTUNITIES
        class CarMock:
            def __init__(self, data):
                self.position = data.position
                self.speed = data.speed
                self.tyre_wear = data.tyre_wear
        
        our_car_mock = CarMock(race_data.our_car)
        
        comp_list = []
        for comp in race_data.competitors:
            comp_mock = type('obj', (object,), {
                'car_id': comp.car_id,
                'position': comp.position,
                'speed': comp.speed,
                'gap': comp.gap,
                'slow_zones': comp.slow_zones,
                'tyre_age': comp.tyre_age
            })()
            comp_list.append(comp_mock)
        
        overtaking_opportunities = strategy_analyzer.find_overtaking_opportunities(
            our_car_mock,
            comp_list,
            race_data.drs_zones
        )
        
        # 4. DETERMINE IMMEDIATE ACTION
        immediate_action = determine_immediate_action(
            pit_recommendation,
            pace_strategy,
            overtaking_opportunities,
            race_data.our_car.current_lap,
            race_data.total_laps
        )
        
        # 5. AUTONOMOUS EXECUTION CHECK
        confidence = immediate_action.get('confidence', 0)
        action = immediate_action.get('action', 'HOLD_POSITION')
        
        if confidence >= 70:
            logger.info(f"🤖 AUTONOMOUS EXECUTION TRIGGERED")
            logger.info(f"   Action: {action}")
            logger.info(f"   Confidence: {confidence}%")
            
            # Execute action
            execution_result = await execute_strategy_action(
                action,
                immediate_action,
                pit_recommendation,
                overtaking_opportunities,
                race_data
            )
            
            # Broadcast to connected clients
            await broadcast_execution(execution_result)
            
            performance_stats['autonomous_executions'] += 1
            
            immediate_action['executed'] = True
            immediate_action['execution_result'] = execution_result
        else:
            logger.info(f"⏸️ Execution skipped: Confidence {confidence}% < 70%")
            immediate_action['executed'] = False
            immediate_action['skip_reason'] = f"Confidence {confidence}% below threshold"
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'lap': race_data.our_car.current_lap,
            'position': race_data.our_car.position,
            
            # IMMEDIATE ACTION with execution status
            'immediate_action': immediate_action,
            
            # Core recommendations
            'pit_stop_recommendation': pit_recommendation,
            'pace_strategy': pace_strategy,
            'overtaking_opportunities': overtaking_opportunities,
            
            # Metadata
            'response_time_ms': round(response_time * 1000, 1),
            'telemetry_samples': len(telemetry_history[car_id]),
            'autonomous_mode': True,
            'execution_threshold': 70
        }
        
        return convert_numpy_types(response)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

async def execute_strategy_action(action: str, immediate_action: Dict,
                                  pit_rec: Dict, overtakes: List[Dict],
                                  race_data: RaceData) -> Dict:
    """Execute the strategy action"""
    
    execution = {
        'action': action,
        'timestamp': datetime.now().isoformat(),
        'lap': race_data.our_car.current_lap,
        'confidence': immediate_action.get('confidence', 0)
    }
    
    if action == 'PIT_NOW':
        execution['details'] = {
            'type': 'pit_stop',
            'compound': pit_rec.get('tyre_compound', 'MEDIUM'),
            'expected_impact': pit_rec.get('expected_time_impact', 0),
            'reasoning': pit_rec.get('reasoning', '')
        }
        logger.info(f"   🔧 Executing pit stop: {execution['details']['compound']}")
        
    elif action == 'OVERTAKE':
        if overtakes:
            target = overtakes[0]
            execution['details'] = {
                'type': 'overtake',
                'target': target.get('target_car', 'unknown'),
                'probability': target.get('probability', 0),
                'zone': target.get('recommended_zone', 'DRS Zone 1')
            }
            logger.info(f"   🏁 Executing overtake on {execution['details']['target']}")
    
    elif action in ['PUSH_PACE', 'CONSERVE']:
        execution['details'] = {
            'type': 'pace_change',
            'mode': action.replace('_PACE', '').replace('CONSERVE', 'CONSERVE'),
            'reasoning': immediate_action.get('reasoning', '')
        }
        logger.info(f"   ⚡ Changing pace to {execution['details']['mode']}")
    
    elif action == 'DEFEND':
        execution['details'] = {
            'type': 'defensive',
            'reasoning': immediate_action.get('reasoning', '')
        }
        logger.info(f"   🛡️ Activating defensive mode")
    
    return execution

def determine_immediate_action(pit_rec: Dict, pace: Dict, overtakes: List[Dict],
                               current_lap: int, total_laps: int) -> Dict:
    """Determine what action to take RIGHT NOW"""
    actions = []
    
    # Critical pit needed?
    if 'URGENT' in pit_rec.get('recommendation', ''):
        return {
            'action': 'PIT_NOW',
            'priority': 'CRITICAL',
            'reasoning': pit_rec['reasoning'],
            'confidence': pit_rec['confidence'],
            'execute_immediately': True
        }
    
    # Overtake opportunity?
    if overtakes and overtakes[0]['probability'] > 65:
        actions.append({
            'action': 'OVERTAKE',
            'priority': 'HIGH',
            'target': overtakes[0]['target_car'],
            'probability': overtakes[0]['probability'],
            'reasoning': overtakes[0]['recommendation'],
            'confidence': overtakes[0]['probability'],
            'execute_immediately': True
        })
    
    # Pit soon?
    if pit_rec.get('laps_until_pit', 999) <= 3:
        actions.append({
            'action': 'PIT_SOON',
            'priority': 'MEDIUM',
            'optimal_lap': pit_rec.get('optimal_lap'),
            'reasoning': pit_rec.get('reasoning'),
            'confidence': pit_rec.get('confidence', 0),
            'execute_immediately': False
        })
    
    # Pace adjustment?
    pace_mode = pace.get('pace_mode', 'BALANCED')
    if pace_mode in ['ATTACK', 'PUSH']:
        actions.append({
            'action': 'PUSH_PACE',
            'priority': 'MEDIUM',
            'reasoning': f"Pace mode: {pace_mode}. {pace.get('recommendation', '')}",
            'confidence': pace.get('confidence', 70),
            'execute_immediately': True
        })
    elif pace_mode == 'CONSERVE':
        actions.append({
            'action': 'CONSERVE',
            'priority': 'MEDIUM',
            'reasoning': pace.get('recommendation', ''),
            'confidence': pace.get('confidence', 70),
            'execute_immediately': True
        })
    
    # Default: hold position
    if not actions:
        return {
            'action': 'HOLD_POSITION',
            'priority': 'LOW',
            'reasoning': 'Maintain current pace and position',
            'confidence': 85,
            'execute_immediately': False
        }
    
    # Return highest priority action
    return actions[0]

# ==================== AUTONOMOUS CONTROL ENDPOINTS ====================

@app.post("/api/autonomous/start")
async def start_autonomous_execution():
    """Start autonomous strategy execution"""
    if not autonomous_executor.running:
        asyncio.create_task(autonomous_executor.start())
        return {
            "status": "started",
            "message": "Autonomous execution activated",
            "threshold": autonomous_executor.min_confidence_threshold
        }
    return {"status": "already_running"}

@app.post("/api/autonomous/stop")
async def stop_autonomous_execution():
    """Stop autonomous strategy execution"""
    autonomous_executor.stop()
    return {
        "status": "stopped",
        "stats": autonomous_executor.get_stats()
    }

@app.get("/api/autonomous/status")
async def get_autonomous_status():
    """Get autonomous execution status"""
    return {
        "running": autonomous_executor.running,
        "stats": autonomous_executor.get_stats(),
        "threshold": autonomous_executor.min_confidence_threshold,
        "analysis_interval": autonomous_executor.analysis_interval
    }

@app.post("/api/autonomous/threshold")
async def set_confidence_threshold(threshold: float):
    """Set confidence threshold for execution"""
    if 0 <= threshold <= 100:
        autonomous_executor.min_confidence_threshold = threshold
        return {
            "status": "updated",
            "new_threshold": threshold
        }
    raise HTTPException(status_code=400, detail="Threshold must be 0-100")

# ==================== UTILITY ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "service": "F1 Quantum Strategy API - Autonomous",
        "version": "4.0.0",
        "status": "operational",
        "autonomous_mode": autonomous_executor.running,
        "features": [
            "✅ Real-time strategy predictions",
            "✅ AUTONOMOUS execution (confidence > 70%)",
            "✅ WebSocket telemetry streaming",
            "✅ Live strategy updates",
            "✅ Quantum-powered decision making"
        ],
        "endpoints": {
            "analyze": "/api/strategy/analyze",
            "websocket": "/ws/telemetry",
            "autonomous_start": "/api/autonomous/start",
            "autonomous_stop": "/api/autonomous/stop",
            "autonomous_status": "/api/autonomous/status"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "autonomous_mode": autonomous_executor.running,
        "engines_loaded": {
            "quantum_strategy": True,
            "strategy_analyzer": True,
            "pit_predictor": True,
            "tyre_model": True,
            "weak_point_detector": True,
            "autonomous_executor": True
        },
        "telemetry_active": len(telemetry_history) > 0,
        "cars_tracked": len(telemetry_history),
        "autonomous_executions": performance_stats['autonomous_executions']
    }

@app.get("/api/stats")
async def get_statistics():
    return convert_numpy_types(performance_stats)

@app.post("/api/reset")
async def reset_telemetry():
    """Reset telemetry history"""
    global telemetry_history
    telemetry_history = {}
    performance_stats['cars_tracked'] = 0
    performance_stats['total_samples'] = 0
    performance_stats['autonomous_executions'] = 0
    return {"status": "Telemetry reset", "timestamp": datetime.now().isoformat()}

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Start autonomous executor on startup"""
    logger.info("\n" + "="*70)
    logger.info("🏎️  F1 QUANTUM STRATEGY BACKEND - AUTONOMOUS v4.0")
    logger.info("="*70)
    logger.info("\n✅ All engines initialized")
    logger.info("✅ Autonomous execution ready")
    logger.info("✅ WebSocket streaming enabled")
    logger.info(f"\n🔗 Server: http://localhost:8000")
    logger.info(f"📚 Docs: http://localhost:8000/docs")
    logger.info(f"📡 WebSocket: ws://localhost:8000/ws/telemetry")
    logger.info("\n" + "="*70 + "\n")
    
    # Auto-start autonomous execution
    asyncio.create_task(autonomous_executor.start())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
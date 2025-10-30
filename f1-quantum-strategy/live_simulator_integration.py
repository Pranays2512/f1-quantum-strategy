"""
Live Simulator Integration
Real-time connection to F1 simulators for continuous strategy updates
"""

import asyncio
import websockets
import json
from typing import Dict, List, Callable, Optional
from datetime import datetime
import threading
from collections import deque

class SimulatorConnector:
    """
    Connect to F1 simulator and stream telemetry in real-time
    
    Supports multiple simulator types:
    - F1 2023/2024 game UDP telemetry
    - Assetto Corsa Competizione
    - iRacing
    - Custom telemetry systems
    """
    
    def __init__(self, strategy_engine, host: str = "localhost", port: int = 20777):
        self.host = host
        self.port = port
        self.strategy_engine = strategy_engine
        
        # Connection state
        self.connected = False
        self.running = False
        
        # Data buffers
        self.telemetry_buffer = deque(maxlen=100)
        self.lap_history = []
        self.current_lap = 0
        
        # Callbacks for events
        self.callbacks = {
            'lap_complete': [],
            'sector_complete': [],
            'pit_entry': [],
            'pit_exit': [],
            'overtake_opportunity': [],
            'strategy_update': []
        }
        
        # Strategy update frequency
        self.update_interval = 1.0  # Update strategy every second
        self.last_strategy_update = None
    
    async def connect(self):
        """Establish connection to simulator"""
        print(f"🔌 Connecting to simulator at {self.host}:{self.port}...")
        
        try:
            # For WebSocket-based simulators
            self.websocket = await websockets.connect(
                f"ws://{self.host}:{self.port}"
            )
            self.connected = True
            print("✅ Connected to simulator!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def start_streaming(self):
        """Start receiving live telemetry"""
        if not self.connected:
            if not await self.connect():
                return
        
        self.running = True
        print("📡 Streaming telemetry...")
        
        try:
            while self.running:
                # Receive telemetry packet
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=5.0
                )
                
                # Parse telemetry
                telemetry = json.loads(message)
                
                # Process telemetry
                await self._process_telemetry(telemetry)
                
        except asyncio.TimeoutError:
            print("⚠️ No data received (timeout)")
        except Exception as e:
            print(f"❌ Streaming error: {e}")
        finally:
            self.running = False
    
    async def _process_telemetry(self, telemetry: Dict):
        """Process incoming telemetry and trigger strategy updates"""
        
        # Extract key data
        current_lap = telemetry.get('currentLap', 0)
        sector = telemetry.get('sector', 0)
        
        # Store in buffer
        self.telemetry_buffer.append({
            'timestamp': datetime.now().isoformat(),
            'data': telemetry
        })
        
        # Detect lap completion
        if current_lap > self.current_lap:
            self.current_lap = current_lap
            await self._on_lap_complete(telemetry)
        
        # Detect sector completion
        if 'sectorTime' in telemetry:
            await self._on_sector_complete(telemetry, sector)
        
        # Periodic strategy update
        await self._update_strategy_if_needed(telemetry)
        
        # Detect special events
        await self._detect_events(telemetry)
    
    async def _on_lap_complete(self, telemetry: Dict):
        """Handle lap completion event"""
        lap_data = self._extract_lap_summary(telemetry)
        self.lap_history.append(lap_data)
        
        print(f"\n🏁 LAP {self.current_lap} COMPLETE")
        print(f"   Time: {lap_data['lap_time']:.3f}s")
        print(f"   Position: P{lap_data['position']}")
        print(f"   Tyre Wear: {lap_data['tyre_wear']:.1f}%")
        
        # Trigger lap complete callbacks
        for callback in self.callbacks['lap_complete']:
            await callback(lap_data)
        
        # Force strategy update
        await self._update_strategy_if_needed(telemetry, force=True)
    
    async def _on_sector_complete(self, telemetry: Dict, sector: int):
        """Handle sector completion"""
        sector_time = telemetry.get('sectorTime', 0)
        
        for callback in self.callbacks['sector_complete']:
            await callback({'sector': sector, 'time': sector_time})
    
    async def _update_strategy_if_needed(self, telemetry: Dict, force: bool = False):
        """Update strategy if interval has passed"""
        now = datetime.now()
        
        if not force and self.last_strategy_update:
            elapsed = (now - self.last_strategy_update).total_seconds()
            if elapsed < self.update_interval:
                return
        
        self.last_strategy_update = now
        
        # Build race state from telemetry
        race_state = self._build_race_state(telemetry)
        
        # Get strategy decision from integrated engine
        decision = self.strategy_engine.decide_optimal_strategy(
            race_state,
            list(self.telemetry_buffer)
        )
        
        # Trigger strategy update callbacks
        for callback in self.callbacks['strategy_update']:
            await callback(decision)
        
        # Display decision
        if decision.should_execute_now:
            print(f"\n⚡ STRATEGY UPDATE:")
            print(f"   Action: {decision.primary_action.value}")
            print(f"   Confidence: {decision.confidence*100:.1f}%")
            print(f"   Expected Position: P{decision.expected_position:.1f}")
            print(f"   Risk Level: {decision.risk_level}")
    
    async def _detect_events(self, telemetry: Dict):
        """Detect special racing events"""
        
        # Detect pit entry/exit
        pit_status = telemetry.get('pitStatus', 0)
        if pit_status == 1:  # Entering pit
            for callback in self.callbacks['pit_entry']:
                await callback(telemetry)
        elif pit_status == 2:  # Exiting pit
            for callback in self.callbacks['pit_exit']:
                await callback(telemetry)
        
        # Detect overtaking opportunity
        gap_ahead = telemetry.get('gapAhead', 10)
        if gap_ahead < 1.5:  # Within striking distance
            for callback in self.callbacks['overtake_opportunity']:
                await callback({
                    'gap': gap_ahead,
                    'drs_available': telemetry.get('drsActive', False)
                })
    
    def _extract_lap_summary(self, telemetry: Dict) -> Dict:
        """Extract lap summary from telemetry"""
        return {
            'lap': self.current_lap,
            'lap_time': telemetry.get('lastLapTime', 0),
            'position': telemetry.get('position', 0),
            'tyre_wear': telemetry.get('tyreWear', [0, 0, 0, 0])[0],  # FL tyre
            'tyre_temp': {
                'FL': telemetry.get('tyreTemp', [0, 0, 0, 0])[0],
                'FR': telemetry.get('tyreTemp', [0, 0, 0, 0])[1],
                'RL': telemetry.get('tyreTemp', [0, 0, 0, 0])[2],
                'RR': telemetry.get('tyreTemp', [0, 0, 0, 0])[3]
            },
            'fuel': telemetry.get('fuelRemaining', 0),
            'sector_times': telemetry.get('sectorTimes', [0, 0, 0])
        }
    
    def _build_race_state(self, telemetry: Dict) -> 'RaceState':
        """Build RaceState object from telemetry"""
        from strategy_feedback_loop import RaceState
        
        return RaceState(
            lap=self.current_lap,
            position=telemetry.get('position', 10),
            tyre_wear=telemetry.get('tyreWear', [50])[0],
            tyre_temps={
                'FL': telemetry.get('tyreTemp', [95, 95, 95, 95])[0],
                'FR': telemetry.get('tyreTemp', [95, 95, 95, 95])[1],
                'RL': telemetry.get('tyreTemp', [95, 95, 95, 95])[2],
                'RR': telemetry.get('tyreTemp', [95, 95, 95, 95])[3]
            },
            fuel=telemetry.get('fuelRemaining', 50),
            gap_ahead=telemetry.get('gapAhead', 5.0),
            gap_behind=telemetry.get('gapBehind', 5.0),
            sector_times=telemetry.get('sectorTimes', [30, 32, 28]),
            competitors=self._extract_competitors(telemetry),
            track_conditions={
                'temperature': telemetry.get('trackTemp', 25),
                'rainfall': telemetry.get('rainIntensity', 0),
                'track_evolution': 85
            },
            total_laps=telemetry.get('totalLaps', 50)
        )
    
    def _extract_competitors(self, telemetry: Dict) -> List[Dict]:
        """Extract competitor data"""
        competitors = []
        
        # Parse competitor array if available
        comp_data = telemetry.get('competitors', [])
        for comp in comp_data:
            competitors.append({
                'car_id': comp.get('id', 'unknown'),
                'position': comp.get('position', 0),
                'gap': comp.get('gap', 0),
                'tyre_age': comp.get('tyreAge', 0)
            })
        
        return competitors
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for specific event"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def stop(self):
        """Stop streaming"""
        self.running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        print("🛑 Stopped streaming")


class SimulatorBridge:
    """
    Bridge between simulator and strategy system
    
    Handles different simulator APIs and normalizes data
    """
    
    SIMULATOR_TYPES = {
        'F1_2024': 'f1_2024',
        'ASSETTO_CORSA': 'ac',
        'IRACING': 'iracing',
        'CUSTOM': 'custom'
    }
    
    def __init__(self, simulator_type: str, strategy_engine):
        self.simulator_type = simulator_type
        self.connector = SimulatorConnector(strategy_engine)
        
        # Setup simulator-specific handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup event handlers based on simulator type"""
        
        async def on_lap_complete(lap_data):
            print(f"📊 Lap {lap_data['lap']}: {lap_data['lap_time']:.3f}s")
        
        async def on_overtake_opportunity(event):
            print(f"🎯 OVERTAKE OPPORTUNITY! Gap: {event['gap']:.2f}s")
        
        async def on_strategy_update(decision):
            if decision.confidence > 0.8:
                print(f"💡 HIGH CONFIDENCE: {decision.primary_action.value}")
        
        self.connector.register_callback('lap_complete', on_lap_complete)
        self.connector.register_callback('overtake_opportunity', on_overtake_opportunity)
        self.connector.register_callback('strategy_update', on_strategy_update)
    
    async def start(self):
        """Start bridge"""
        print(f"🌉 Starting bridge for {self.simulator_type}...")
        await self.connector.start_streaming()
    
    def stop(self):
        """Stop bridge"""
        self.connector.stop()


# Mock simulator for testing
class MockSimulator:
    """Simulates F1 telemetry for testing"""
    
    def __init__(self, host: str = "localhost", port: int = 20777):
        self.host = host
        self.port = port
        self.current_lap = 1
        self.running = False
    
    async def start(self):
        """Start mock simulator server"""
        self.running = True
        
        async def handler(websocket, path):
            print("📱 Client connected to mock simulator")
            
            while self.running:
                # Generate fake telemetry
                telemetry = self._generate_telemetry()
                
                # Send to client
                await websocket.send(json.dumps(telemetry))
                
                # Update state
                self.current_lap += 0.01  # Slowly increment
                
                # Wait before next packet
                await asyncio.sleep(0.1)
        
        async with websockets.serve(handler, self.host, self.port):
            print(f"🎮 Mock simulator running on {self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    def _generate_telemetry(self) -> Dict:
        """Generate realistic telemetry data"""
        lap_progress = self.current_lap % 1.0
        
        return {
            'currentLap': int(self.current_lap),
            'position': 3,
            'lastLapTime': 89.234 + np.random.normal(0, 0.2),
            'tyreWear': [min(100, self.current_lap * 2.5)] * 4,
            'tyreTemp': [95 + lap_progress * 10] * 4,
            'fuelRemaining': max(0, 110 - self.current_lap * 1.8),
            'gapAhead': 2.5 + np.random.normal(0, 0.3),
            'gapBehind': 3.0 + np.random.normal(0, 0.3),
            'sectorTimes': [28.5, 32.0, 28.7],
            'sector': int(lap_progress * 3),
            'pitStatus': 0,
            'drsActive': lap_progress > 0.7,
            'trackTemp': 28,
            'rainIntensity': 0,
            'totalLaps': 50,
            'competitors': [
                {'id': 'car_2', 'position': 2, 'gap': -2.5, 'tyreAge': int(self.current_lap) + 5},
                {'id': 'car_4', 'position': 4, 'gap': 3.0, 'tyreAge': int(self.current_lap) - 2}
            ]
        }
    
    def stop(self):
        """Stop mock simulator"""
        self.running = False


# Integration example
async def main():
    """Test live simulator integration"""
    from strategy_feedback_loop import IntegratedStrategyEngine
    
    # Initialize strategy system (you'd import your actual modules here)
    # strategy_engine = IntegratedStrategyEngine(...)
    
    print("🏎️ F1 Live Strategy System")
    print("=" * 50)
    
    # Option 1: Connect to real simulator
    # bridge = SimulatorBridge('F1_2024', strategy_engine)
    # await bridge.start()
    
    # Option 2: Use mock simulator for testing
    print("\n🎮 Starting mock simulator...")
    mock_sim = MockSimulator()
    
    # Start mock simulator in background
    simulator_task = asyncio.create_task(mock_sim.start())
    
    # Wait for simulator to start
    await asyncio.sleep(2)
    
    # Connect strategy system
    print("\n🤖 Connecting strategy system...")
    # connector = SimulatorConnector(strategy_engine)
    # await connector.start_streaming()
    
    print("\n✅ System running! Press Ctrl+C to stop")
    
    try:
        await simulator_task
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        mock_sim.stop()


if __name__ == '__main__':
    asyncio.run(main())
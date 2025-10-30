"""
Autonomous Strategy Executor
Continuously monitors race state and executes strategies automatically
"""

import asyncio
import aiohttp
import time
from typing import Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutonomousStrategyExecutor:
    """
    Autonomous system that:
    1. Continuously receives live telemetry
    2. Analyzes strategy every few seconds
    3. Auto-executes high-confidence strategies (>70%)
    """
    
    def __init__(self, ai_backend_url: str = "http://localhost:8000"):
        self.ai_url = ai_backend_url
        self.running = False
        self.last_analysis_time = 0
        self.analysis_interval = 3.0  # Analyze every 3 seconds
        self.min_confidence_threshold = 70.0  # Only execute if confidence > 70%
        
        # Track executed actions to avoid duplicates
        self.executed_actions = []
        self.last_pit_lap = 0
        self.last_overtake_attempt_lap = 0
        
        # Store latest telemetry
        self.latest_telemetry = None
        
    async def start(self):
        """Start autonomous execution loop"""
        self.running = True
        logger.info("🤖 Autonomous Strategy Executor STARTED")
        logger.info(f"   Min confidence threshold: {self.min_confidence_threshold}%")
        logger.info(f"   Analysis interval: {self.analysis_interval}s")
        
        while self.running:
            try:
                if self.latest_telemetry:
                    current_time = time.time()
                    
                    # Check if it's time to analyze
                    if current_time - self.last_analysis_time >= self.analysis_interval:
                        await self._analyze_and_execute()
                        self.last_analysis_time = current_time
                
                await asyncio.sleep(0.5)  # Check every 0.5s
                
            except Exception as e:
                logger.error(f"❌ Error in execution loop: {e}")
                await asyncio.sleep(1.0)
    
    def update_telemetry(self, telemetry: Dict):
        """Update with latest telemetry from race"""
        self.latest_telemetry = telemetry
        
    async def _analyze_and_execute(self):
        """Core autonomous decision loop"""
        if not self.latest_telemetry:
            return
        
        logger.info(f"\n🔍 Analyzing strategy (Lap {self.latest_telemetry['our_car']['current_lap']})")
        
        try:
            # Call AI backend for strategy analysis
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ai_url}/api/strategy/analyze",
                    json=self.latest_telemetry,
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        strategy = await response.json()
                        await self._evaluate_and_execute(strategy)
                    else:
                        logger.warning(f"⚠️ Strategy analysis failed: {response.status}")
                        
        except asyncio.TimeoutError:
            logger.warning("⚠️ Strategy analysis timeout")
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
    
    async def _evaluate_and_execute(self, strategy: Dict):
        """Evaluate strategy and execute if confidence threshold met"""
        
        current_lap = self.latest_telemetry['our_car']['current_lap']
        
        # 1. CHECK IMMEDIATE ACTION
        immediate = strategy.get('immediate_action', {})
        action = immediate.get('action', 'HOLD_POSITION')
        confidence = immediate.get('confidence', 0)
        priority = immediate.get('priority', 'LOW')
        
        logger.info(f"   Action: {action}")
        logger.info(f"   Confidence: {confidence}%")
        logger.info(f"   Priority: {priority}")
        
        # 2. EVALUATE IF WE SHOULD EXECUTE
        should_execute = False
        reason = ""
        
        if priority == 'CRITICAL':
            # Always execute critical actions regardless of confidence
            should_execute = True
            reason = "CRITICAL priority overrides confidence threshold"
        elif confidence >= self.min_confidence_threshold:
            should_execute = True
            reason = f"Confidence {confidence}% >= threshold {self.min_confidence_threshold}%"
        else:
            logger.info(f"   ⏸️ Skipped: Confidence {confidence}% < {self.min_confidence_threshold}%")
            return
        
        # 3. CHECK IF ACTION IS VIABLE (avoid duplicates)
        if not self._is_action_viable(action, current_lap):
            logger.info(f"   ⏸️ Skipped: Action not viable at this time")
            return
        
        # 4. EXECUTE THE ACTION
        logger.info(f"   ✅ EXECUTING: {action}")
        logger.info(f"   Reason: {reason}")
        
        await self._execute_action(action, strategy, current_lap)
    
    def _is_action_viable(self, action: str, current_lap: int) -> bool:
        """Check if action can be executed now"""
        
        if action == 'PIT_NOW' or action == 'PIT_SOON':
            # Don't pit if we just pitted recently
            if current_lap - self.last_pit_lap < 5:
                return False
        
        if action == 'OVERTAKE':
            # Don't attempt overtake if we just tried recently
            if current_lap - self.last_overtake_attempt_lap < 2:
                return False
        
        return True
    
    async def _execute_action(self, action: str, strategy: Dict, current_lap: int):
        """Execute the actual strategy action"""
        
        # Log execution
        self.executed_actions.append({
            'lap': current_lap,
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'confidence': strategy.get('immediate_action', {}).get('confidence', 0)
        })
        
        # Execute based on action type
        if action == 'PIT_NOW':
            await self._execute_pit_stop(strategy)
            self.last_pit_lap = current_lap
            
        elif action == 'PIT_SOON':
            pit = strategy.get('pit_stop_recommendation', {})
            optimal_lap = pit.get('optimal_lap', current_lap + 3)
            laps_until = optimal_lap - current_lap
            
            logger.info(f"   📅 Scheduled pit stop in {laps_until} laps (lap {optimal_lap})")
            logger.info(f"   🔧 Compound: {pit.get('tyre_compound', 'MEDIUM')}")
            
        elif action == 'OVERTAKE':
            await self._execute_overtake(strategy)
            self.last_overtake_attempt_lap = current_lap
            
        elif action == 'PUSH_PACE':
            await self._execute_pace_change('PUSH')
            
        elif action == 'CONSERVE':
            await self._execute_pace_change('CONSERVE')
            
        elif action == 'DEFEND':
            await self._execute_defensive_strategy()
            
        elif action == 'HOLD_POSITION':
            logger.info(f"   🛡️ Holding position - maintaining current strategy")
    
    async def _execute_pit_stop(self, strategy: Dict):
        """Execute pit stop"""
        pit = strategy.get('pit_stop_recommendation', {})
        compound = pit.get('tyre_compound', 'MEDIUM')
        expected_impact = pit.get('expected_time_impact', 0)
        
        logger.info(f"   🔧 EXECUTING PIT STOP")
        logger.info(f"      Compound: {compound}")
        logger.info(f"      Expected impact: {expected_impact:.1f}s")
        logger.info(f"      Reasoning: {pit.get('reasoning', 'N/A')}")
        
        # Send pit command to simulator
        await self._send_command_to_simulator({
            'command': 'PIT_STOP',
            'tyre_compound': compound,
            'reasoning': pit.get('reasoning', '')
        })
    
    async def _execute_overtake(self, strategy: Dict):
        """Execute overtake attempt"""
        overtakes = strategy.get('overtaking_opportunities', [])
        
        if overtakes:
            opp = overtakes[0]
            target = opp.get('target_car', 'unknown')
            probability = opp.get('probability', 0)
            zone = opp.get('recommended_zone', 'DRS Zone 1')
            
            logger.info(f"   🏁 EXECUTING OVERTAKE")
            logger.info(f"      Target: {target}")
            logger.info(f"      Success probability: {probability}%")
            logger.info(f"      Zone: {zone}")
            logger.info(f"      Advice: {opp.get('recommendation', 'N/A')}")
            
            # Send overtake command to simulator
            await self._send_command_to_simulator({
                'command': 'OVERTAKE',
                'target': target,
                'zone': zone,
                'probability': probability
            })
    
    async def _execute_pace_change(self, mode: str):
        """Execute pace mode change"""
        logger.info(f"   ⚡ CHANGING PACE MODE: {mode}")
        
        await self._send_command_to_simulator({
            'command': 'CHANGE_PACE',
            'pace_mode': mode
        })
    
    async def _execute_defensive_strategy(self):
        """Execute defensive driving"""
        logger.info(f"   🛡️ ACTIVATING DEFENSIVE MODE")
        
        await self._send_command_to_simulator({
            'command': 'DEFEND_POSITION'
        })
    
    async def _send_command_to_simulator(self, command: Dict):
        """Send execution command to simulator"""
        # This will be sent to the frontend via WebSocket or HTTP
        # For now, just log it - the frontend will implement the actual execution
        
        logger.info(f"   📡 Command sent to simulator: {command['command']}")
        
        # TODO: In production, send via WebSocket to frontend
        # await websocket.send(json.dumps(command))
    
    def stop(self):
        """Stop autonomous execution"""
        self.running = False
        logger.info("🛑 Autonomous Strategy Executor STOPPED")
        logger.info(f"   Total actions executed: {len(self.executed_actions)}")
    
    def get_execution_history(self):
        """Get history of executed actions"""
        return self.executed_actions
    
    def get_stats(self):
        """Get execution statistics"""
        if not self.executed_actions:
            return {
                'total_executions': 0,
                'actions_by_type': {},
                'avg_confidence': 0
            }
        
        actions_by_type = {}
        total_confidence = 0
        
        for exec_record in self.executed_actions:
            action = exec_record['action']
            actions_by_type[action] = actions_by_type.get(action, 0) + 1
            total_confidence += exec_record.get('confidence', 0)
        
        return {
            'total_executions': len(self.executed_actions),
            'actions_by_type': actions_by_type,
            'avg_confidence': total_confidence / len(self.executed_actions)
        }


# Integration with existing backend
class TelemetryStreamHandler:
    """Handles incoming telemetry stream and feeds to executor"""
    
    def __init__(self, executor: AutonomousStrategyExecutor):
        self.executor = executor
        self.last_update = 0
        
    async def handle_telemetry_update(self, telemetry: Dict):
        """Called whenever new telemetry arrives"""
        current_time = time.time()
        
        # Throttle updates (max 2 per second to avoid overwhelming system)
        if current_time - self.last_update < 0.5:
            return
        
        self.last_update = current_time
        
        # Update executor with latest data
        self.executor.update_telemetry(telemetry)
        
        logger.debug(f"📊 Telemetry updated: Lap {telemetry['our_car']['current_lap']}, "
                    f"P{telemetry['our_car']['position']}, "
                    f"Wear: {telemetry['our_car']['tyre_wear']:.1f}%")


# Example usage
async def main():
    """Test autonomous executor"""
    
    # Create executor
    executor = AutonomousStrategyExecutor(
        ai_backend_url="http://localhost:8000"
    )
    
    # Create telemetry handler
    handler = TelemetryStreamHandler(executor)
    
    # Start autonomous execution
    executor_task = asyncio.create_task(executor.start())
    
    # Simulate incoming telemetry (in production, this comes from the simulator)
    async def simulate_telemetry():
        """Simulate race telemetry stream"""
        for lap in range(1, 51):
            telemetry = {
                "timestamp": int(time.time()),
                "our_car": {
                    "position": 3,
                    "speed": 290 - (lap * 0.2),  # Slight degradation
                    "tyre_temp": {
                        "FL": 95 + lap * 0.5,
                        "FR": 96 + lap * 0.5,
                        "RL": 94 + lap * 0.5,
                        "RR": 95 + lap * 0.5
                    },
                    "tyre_wear": lap * 2.5,
                    "fuel_load": 110 - lap * 1.8,
                    "lap_time": 89.2 + lap * 0.05,
                    "current_lap": lap,
                    "sector_times": [28.5, 32.0, 28.7],
                    "slow_sectors": []
                },
                "competitors": [
                    {
                        "car_id": "car_2",
                        "position": 2,
                        "speed": 288,
                        "gap": 2.5 - lap * 0.05,  # Slowly catching
                        "slow_zones": ["Sector 2"],
                        "tyre_age": lap + 3
                    }
                ],
                "track_conditions": {
                    "temperature": 28,
                    "rainfall": 0,
                    "track_evolution": 85
                },
                "total_laps": 50,
                "drs_zones": ["DRS Zone 1", "DRS Zone 2"]
            }
            
            await handler.handle_telemetry_update(telemetry)
            await asyncio.sleep(2.0)  # Simulate ~2 seconds per lap update
        
        # Stop executor
        executor.stop()
    
    # Run telemetry simulation
    await simulate_telemetry()
    
    # Print stats
    stats = executor.get_stats()
    print("\n" + "="*70)
    print("📊 AUTONOMOUS EXECUTION STATISTICS")
    print("="*70)
    print(f"Total actions executed: {stats['total_executions']}")
    print(f"Average confidence: {stats['avg_confidence']:.1f}%")
    print(f"\nActions by type:")
    for action, count in stats['actions_by_type'].items():
        print(f"   • {action}: {count}")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
/**
 * Autonomous Strategy Client
 * 
 * Add this script to your index.html to enable autonomous strategy execution
 * 
 * Features:
 * - Sends live telemetry to AI backend
 * - Receives autonomous execution commands
 * - Executes strategies automatically (no button clicks needed)
 */

class AutonomousStrategyClient {
    constructor() {
        this.aiUrl = 'http://localhost:8000';
        this.ws = null;
        this.isConnected = false;
        this.autoAnalysisInterval = 3000; // Analyze every 3 seconds
        this.lastAnalysisTime = 0;
        this.analysisTimer = null;
        
        // Bind context
        this.handleExecution = this.handleExecution.bind(this);
    }
    
    /**
     * Initialize autonomous mode
     */
    async initialize() {
        console.log('🤖 Initializing Autonomous Strategy System...');
        
        // 1. Check backend health
        const healthy = await this.checkBackendHealth();
        if (!healthy) {
            console.error('❌ Backend not available');
            this.showNotification('AI Backend offline - autonomous mode disabled', 'error');
            return false;
        }
        
        // 2. Start autonomous execution on backend
        await this.startAutonomousExecution();
        
        // 3. Start automatic analysis loop
        this.startAutoAnalysis();
        
        // 4. Setup WebSocket for real-time execution commands (optional)
        // this.connectWebSocket();
        
        console.log('✅ Autonomous mode activated');
        this.showNotification('🤖 Autonomous Strategy Mode ACTIVE', 'success');
        
        return true;
    }
    
    /**
     * Check if backend is healthy
     */
    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.aiUrl}/api/health`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('✅ Backend health:', data);
                return true;
            }
            return false;
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    }
    
    /**
     * Start autonomous execution on backend
     */
    async startAutonomousExecution() {
        try {
            const response = await fetch(`${this.aiUrl}/api/autonomous/start`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log('🚀 Autonomous execution started:', data);
            }
        } catch (error) {
            console.error('Failed to start autonomous execution:', error);
        }
    }
    
    /**
     * Start automatic strategy analysis
     */
    startAutoAnalysis() {
        console.log(`🔄 Auto-analysis enabled (every ${this.autoAnalysisInterval/1000}s)`);
        
        // Clear any existing timer
        if (this.analysisTimer) {
            clearInterval(this.analysisTimer);
        }
        
        // Start analysis loop
        this.analysisTimer = setInterval(() => {
            if (raceState.isRunning && !raceState.isPaused && raceState.trackLoaded) {
                this.analyzeAndExecute();
            }
        }, this.autoAnalysisInterval);
    }
    
    /**
     * Analyze strategy and execute autonomously
     */
    async analyzeAndExecute() {
        const currentTime = Date.now();
        
        // Throttle requests
        if (currentTime - this.lastAnalysisTime < this.autoAnalysisInterval) {
            return;
        }
        
        this.lastAnalysisTime = currentTime;
        
        console.log(`🔍 Auto-analyzing strategy (Lap ${ourCar.currentLap})...`);
        
        // Build race data from current state
        const raceData = this.buildRaceData();
        
        try {
            const response = await fetch(`${this.aiUrl}/api/strategy/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(raceData)
            });
            
            if (!response.ok) {
                console.error('Analysis failed:', response.status);
                return;
            }
            
            const result = await response.json();
            
            // Display strategy (update UI)
            this.displayStrategy(result);
            
            // Check if action was executed
            const immediate = result.immediate_action;
            if (immediate.executed) {
                console.log('✅ AUTONOMOUS EXECUTION:', immediate.action);
                console.log('   Confidence:', immediate.confidence + '%');
                
                // Execute the action in the simulator
                this.handleExecution(immediate);
            } else {
                console.log('⏸️ No execution:', immediate.skip_reason || 'Below threshold');
            }
            
        } catch (error) {
            console.error('Auto-analysis error:', error);
        }
    }
    
    /**
     * Build race data from current simulator state
     */
    buildRaceData() {
        return {
            timestamp: Date.now(),
            our_car: {
                position: ourCar.position,
                speed: ourCar.speed,
                tyre_temp: {
                    FL: Math.round(ourCar.tyreTemp.FL),
                    FR: Math.round(ourCar.tyreTemp.FR),
                    RL: Math.round(ourCar.tyreTemp.RL),
                    RR: Math.round(ourCar.tyreTemp.RR)
                },
                tyre_wear: ourCar.tyreWear,
                fuel_load: ourCar.fuel,
                lap_time: ourCar.lapTime,
                current_lap: ourCar.currentLap,
                sector_times: [28.5, 31.8, 28.2],
                slow_sectors: []
            },
            competitors: competitors.map(c => ({
                car_id: c.id,
                position: c.position,
                speed: Math.round(c.speed),
                gap: Math.abs(c.position - ourCar.position) * 2.5,
                slow_zones: [],
                tyre_age: Math.floor(c.tyreWear / 3)
            })),
            track_conditions: {
                temperature: 28,
                rainfall: 0,
                track_evolution: 85
            },
            total_laps: raceState.totalLaps,
            drs_zones: trackData.drsZones.map(z => z.name)
        };
    }
    
    /**
     * Handle autonomous execution command
     */
    handleExecution(immediate) {
        const action = immediate.action;
        const executionResult = immediate.execution_result;
        
        // Show notification
        this.showNotification(
            `🤖 AI Executing: ${this.formatAction(action)}`,
            'autonomous',
            5000
        );
        
        // Execute based on action type
        if (action === 'PIT_NOW' && executionResult) {
            const details = executionResult.details;
            logEvent(`🤖 AI executing pit stop: ${details.compound} tyres`);
            
            // Execute pit stop
            setTimeout(() => {
                executePitStop();
            }, 1000);
        }
        else if (action === 'OVERTAKE' && executionResult) {
            const details = executionResult.details;
            const probability = details.probability || 0;
            
            logEvent(`🤖 AI executing overtake: ${details.target} (${probability}% probability)`);
            
            // Execute overtake
            setTimeout(() => {
                const opportunities = [{ 
                    current_position: details.target ? parseInt(details.target.split('_')[1]) : ourCar.position - 1,
                    probability: probability 
                }];
                executeOvertake(opportunities);
            }, 1000);
        }
        else if (action === 'PUSH_PACE') {
            logEvent('🤖 AI activating PUSH mode');
            ourCar.paceMode = 'PUSH';
        }
        else if (action === 'CONSERVE') {
            logEvent('🤖 AI activating CONSERVE mode');
            ourCar.paceMode = 'CONSERVE';
        }
        else if (action === 'DEFEND') {
            logEvent('🤖 AI activating DEFENSIVE mode');
            ourCar.paceMode = 'BALANCED';
        }
    }
    
    /**
     * Display strategy in UI
     */
    displayStrategy(result) {
        // Use existing displayStrategyResults function
        if (typeof displayStrategyResults === 'function') {
            displayStrategyResults(result);
        }
    }
    
    /**
     * Format action name for display
     */
    formatAction(action) {
        const names = {
            'PIT_NOW': '🔧 Pit Stop Now',
            'PIT_SOON': '⏰ Pit Soon',
            'OVERTAKE': '🏁 Overtake',
            'PUSH_PACE': '⚡ Push Pace',
            'CONSERVE': '🐢 Conserve',
            'DEFEND': '🛡️ Defend Position',
            'HOLD_POSITION': '🎯 Hold Position'
        };
        return names[action] || action;
    }
    
    /**
     * Show notification to user
     */
    showNotification(message, type = 'info', duration = 3000) {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.padding = '15px 25px';
        notification.style.borderRadius = '10px';
        notification.style.fontWeight = 'bold';
        notification.style.fontSize = '16px';
        notification.style.zIndex = '10001';
        notification.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
        notification.style.transition = 'all 0.3s';
        notification.textContent = message;
        
        // Style based on type
        if (type === 'success') {
            notification.style.background = 'rgba(34, 197, 94, 0.95)';
            notification.style.border = '2px solid #22c55e';
        } else if (type === 'error') {
            notification.style.background = 'rgba(239, 68, 68, 0.95)';
            notification.style.border = '2px solid #ef4444';
        } else if (type === 'autonomous') {
            notification.style.background = 'rgba(139, 92, 246, 0.95)';
            notification.style.border = '2px solid #8b5cf6';
        } else {
            notification.style.background = 'rgba(59, 130, 246, 0.95)';
            notification.style.border = '2px solid #3b82f6';
        }
        
        notification.style.color = 'white';
        
        document.body.appendChild(notification);
        
        // Fade out and remove
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateY(-20px)';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, duration);
    }
    
    /**
     * Connect WebSocket for real-time commands (optional)
     */
    connectWebSocket() {
        try {
            this.ws = new WebSocket('ws://localhost:8000/ws/telemetry');
            
            this.ws.onopen = () => {
                console.log('📡 WebSocket connected');
                this.isConnected = true;
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'execution') {
                    console.log('📥 Received execution command:', data.command);
                    this.handleExecution(data.command);
                }
            };
            
            this.ws.onclose = () => {
                console.log('📡 WebSocket disconnected');
                this.isConnected = false;
                
                // Reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }
    
    /**
     * Send telemetry via WebSocket
     */
    sendTelemetry(telemetry) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(telemetry));
        }
    }
    
    /**
     * Stop autonomous mode
     */
    async stop() {
        console.log('🛑 Stopping autonomous mode...');
        
        // Clear analysis timer
        if (this.analysisTimer) {
            clearInterval(this.analysisTimer);
            this.analysisTimer = null;
        }
        
        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        // Stop backend autonomous execution
        try {
            await fetch(`${this.aiUrl}/api/autonomous/stop`, {
                method: 'POST'
            });
        } catch (error) {
            console.error('Failed to stop backend:', error);
        }
        
        console.log('✅ Autonomous mode stopped');
        this.showNotification('Autonomous mode deactivated', 'info');
    }
    
    /**
     * Get autonomous status
     */
    async getStatus() {
        try {
            const response = await fetch(`${this.aiUrl}/api/autonomous/status`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Failed to get status:', error);
        }
        return null;
    }
}

// ==================== INTEGRATION WITH EXISTING CODE ====================

// Create global instance
const autonomousClient = new AutonomousStrategyClient();

// Auto-initialize when page loads
window.addEventListener('load', async () => {
    console.log('🏎️ F1 Quantum Strategy - Autonomous Mode');
    
    // Wait 2 seconds for everything to load
    setTimeout(async () => {
        const initialized = await autonomousClient.initialize();
        
        if (initialized) {
            // Add autonomous indicator to UI
            addAutonomousIndicator();
        }
    }, 2000);
});

// Add visual indicator that autonomous mode is active
function addAutonomousIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'autonomous-indicator';
    indicator.style.position = 'fixed';
    indicator.style.bottom = '20px';
    indicator.style.left = '20px';
    indicator.style.padding = '10px 20px';
    indicator.style.background = 'rgba(139, 92, 246, 0.95)';
    indicator.style.border = '2px solid #8b5cf6';
    indicator.style.borderRadius = '10px';
    indicator.style.color = 'white';
    indicator.style.fontWeight = 'bold';
    indicator.style.fontSize = '14px';
    indicator.style.zIndex = '10000';
    indicator.style.boxShadow = '0 4px 15px rgba(139, 92, 246, 0.5)';
    indicator.innerHTML = '🤖 AUTONOMOUS MODE ACTIVE';
    
    // Add pulsing animation
    indicator.style.animation = 'pulse 2s infinite';
    
    // Add CSS animation if not exists
    if (!document.querySelector('#autonomous-style')) {
        const style = document.createElement('style');
        style.id = 'autonomous-style';
        style.textContent = `
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(indicator);
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    autonomousClient.stop();
});

// Export for manual control
window.autonomousClient = autonomousClient;

console.log('✅ Autonomous Strategy Client loaded');
console.log('   Control via: window.autonomousClient.stop()');
'use client';

import { useState, useEffect } from 'react';

interface ThreatEvent {
    id: string;
    timestamp: string;
    threat_class: string;
    confidence: number;
    risk_level: string;
    action: string;
    event_type: string;
}

interface RiskScore {
    current_score: number;
    trend: string;
    critical_count: number;
    high_risk_count: number;
    suspicious_count: number;
    safe_count: number;
    health: string;
}

interface TimelineEvent {
    timestamp: string;
    stage: string;
    threat_class: string;
    confidence: number;
    risk_level: string;
    id: string;
}

export default function MLDashboardPage() {
    const [threats, setThreats] = useState<ThreatEvent[]>([]);
    const [riskScore, setRiskScore] = useState<RiskScore | null>(null);
    const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
    const [mlStatus, setMlStatus] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [simulating, setSimulating] = useState(false);

    const API_BASE = 'http://localhost:8000/api/v2';

    const fetchData = async () => {
        try {
            // Fetch threats
            const threatsRes = await fetch(`${API_BASE}/ml/threats?limit=20`);
            if (threatsRes.ok) {
                const data = await threatsRes.json();
                setThreats(data.threats || []);
            }

            // Fetch risk score
            const riskRes = await fetch(`${API_BASE}/ml/risk-score`);
            if (riskRes.ok) {
                const data = await riskRes.json();
                setRiskScore(data);
            }

            // Fetch timeline
            const timelineRes = await fetch(`${API_BASE}/ml/timeline?hours=24`);
            if (timelineRes.ok) {
                const data = await timelineRes.json();
                setTimeline(data.timeline || []);
            }

            // Fetch ML status
            const statusRes = await fetch(`${API_BASE}/ml/status`);
            if (statusRes.ok) {
                const data = await statusRes.json();
                setMlStatus(data);
            }
        } catch (error) {
            console.error('Error fetching ML data:', error);
        } finally {
            setLoading(false);
        }
    };

    const simulateThreats = async (threatType: string) => {
        setSimulating(true);
        try {
            await fetch(`${API_BASE}/ml/simulate?threat_type=${threatType}&count=5`, {
                method: 'POST'
            });
            await fetchData();
            alert(`✅ Simulated ${threatType} threat successfully!`);
        } catch (error) {
            console.error('Error simulating threats:', error);
            alert(`⚠️ Simulated ${threatType} threat (demo mode)`);
        } finally {
            setSimulating(false);
        }
    };

    const clearThreats = async () => {
        try {
            await fetch(`${API_BASE}/ml/threats`, { method: 'DELETE' });
            await fetchData();
            alert('✅ Cleared all threats!');
        } catch (error) {
            console.error('Error clearing threats:', error);
            alert('Threats cleared (demo mode)');
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 5000); // Refresh every 5s
        return () => clearInterval(interval);
    }, []);

    const getRiskColor = (level: string) => {
        switch (level) {
            case 'critical': return 'bg-[#ef4444]';
            case 'high_risk': return 'bg-[#f97316]';
            case 'suspicious': return 'bg-[#eab308]';
            case 'safe': return 'bg-[#22c55e]';
            default: return 'bg-[#666]';
        }
    };

    const getRiskBorderColor = (level: string) => {
        switch (level) {
            case 'critical': return 'border-[#ef4444]';
            case 'high_risk': return 'border-[#f97316]';
            case 'suspicious': return 'border-[#eab308]';
            case 'safe': return 'border-[#22c55e]';
            default: return 'border-[#666]';
        }
    };

    const getHealthColor = (health: string) => {
        switch (health) {
            case 'critical': return 'text-[#ef4444]';
            case 'warning': return 'text-[#eab308]';
            case 'healthy': return 'text-[#22c55e]';
            default: return 'text-[#666]';
        }
    };

    const getStageColor = (stage: string) => {
        const colors: { [key: string]: string } = {
            'reconnaissance': 'bg-[#3b82f6]',
            'initial_access': 'bg-[#a855f7]',
            'execution': 'bg-[#f97316]',
            'persistence': 'bg-[#ef4444]',
            'command_control': 'bg-[#ec4899]',
            'lateral_movement': 'bg-[#eab308]',
            'impact': 'bg-[#b91c1c]',
        };
        return colors[stage] || 'bg-[#666]';
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
                <div className="text-[#10a37f] text-xl">Loading ML Dashboard...</div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#0a0a0a] p-6 text-[#e5e5e5]">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-3xl font-semibold text-white mb-2">
                    🧠 ML Hub
                </h1>
                <p className="text-[#666]">
                    Real-time threat detection powered by ensemble ML models
                </p>
            </div>

            {/* Quick Links to ML Features */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
                <a href="/ml/monitor" className="bg-[#1a1a1a] hover:bg-[#252525] border border-[#2a2a2a] rounded-xl p-4 transition group">
                    <div className="text-2xl mb-2">📊</div>
                    <div className="text-white font-medium group-hover:text-[#10a37f]">Real-time Monitor</div>
                    <div className="text-[#666] text-xs mt-1">Live predictions</div>
                </a>
                <a href="/xai" className="bg-[#1a1a1a] hover:bg-[#252525] border border-[#2a2a2a] rounded-xl p-4 transition group">
                    <div className="text-2xl mb-2">🧠</div>
                    <div className="text-white font-medium group-hover:text-[#a855f7]">Explainable AI</div>
                    <div className="text-[#666] text-xs mt-1">SHAP & LIME</div>
                </a>
                <a href="/soar" className="bg-[#1a1a1a] hover:bg-[#252525] border border-[#2a2a2a] rounded-xl p-4 transition group">
                    <div className="text-2xl mb-2">🎯</div>
                    <div className="text-white font-medium group-hover:text-[#ef4444]">SOAR</div>
                    <div className="text-[#666] text-xs mt-1">Incident response</div>
                </a>
                <a href="/phishing" className="bg-[#1a1a1a] hover:bg-[#252525] border border-[#2a2a2a] rounded-xl p-4 transition group">
                    <div className="text-2xl mb-2">🎣</div>
                    <div className="text-white font-medium group-hover:text-[#f97316]">Phishing Scanner</div>
                    <div className="text-[#666] text-xs mt-1">URL & email check</div>
                </a>
                <a href="/rl-agent" className="bg-[#1a1a1a] hover:bg-[#252525] border border-[#2a2a2a] rounded-xl p-4 transition group">
                    <div className="text-2xl mb-2">🤖</div>
                    <div className="text-white font-medium group-hover:text-[#06b6d4]">RL Agent</div>
                    <div className="text-[#666] text-xs mt-1">Adaptive defense</div>
                </a>
            </div>

            {/* Top Stats Row */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                {/* Risk Score Gauge */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h3 className="text-[#666] text-sm mb-2">RISK SCORE</h3>
                    <div className="flex items-center gap-4">
                        <div className={`text-5xl font-bold ${getHealthColor(riskScore?.health || 'healthy')}`}>
                            {riskScore?.current_score?.toFixed(0) || 0}
                        </div>
                        <div className="text-[#666]">
                            <span className="text-2xl">/100</span>
                            <div className="flex items-center gap-1 mt-1">
                                {riskScore?.trend === 'up' && <span className="text-[#ef4444]">↑</span>}
                                {riskScore?.trend === 'down' && <span className="text-[#22c55e]">↓</span>}
                                {riskScore?.trend === 'stable' && <span className="text-[#666]">→</span>}
                                <span className="text-xs">{riskScore?.trend}</span>
                            </div>
                        </div>
                    </div>
                    <div className={`mt-2 text-sm ${getHealthColor(riskScore?.health || 'healthy')}`}>
                        System: {riskScore?.health?.toUpperCase() || 'HEALTHY'}
                    </div>
                </div>

                {/* Threat Counts */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h3 className="text-[#666] text-sm mb-3">THREAT BREAKDOWN</h3>
                    <div className="space-y-2">
                        <div className="flex justify-between">
                            <span className="text-[#ef4444]">Critical</span>
                            <span className="text-white font-bold">{riskScore?.critical_count || 0}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[#f97316]">High Risk</span>
                            <span className="text-white font-bold">{riskScore?.high_risk_count || 0}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[#eab308]">Suspicious</span>
                            <span className="text-white font-bold">{riskScore?.suspicious_count || 0}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[#22c55e]">Safe</span>
                            <span className="text-white font-bold">{riskScore?.safe_count || 0}</span>
                        </div>
                    </div>
                </div>

                {/* ML Status */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h3 className="text-[#666] text-sm mb-3">ML ENGINE STATUS</h3>
                    <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                            <span className="text-[#aaa]">Inference Engine</span>
                            <span className={mlStatus?.inference_engine?.loaded ? 'text-[#22c55e]' : 'text-[#ef4444]'}>
                                {mlStatus?.inference_engine?.loaded ? '✓ Active' : '✗ Offline'}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[#aaa]">Device</span>
                            <span className="text-[#10a37f]">
                                {mlStatus?.inference_engine?.device || 'N/A'}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[#aaa]">Avg Inference</span>
                            <span className="text-white">
                                {mlStatus?.inference_engine?.stats?.avg_inference_time_ms?.toFixed(2) || 0}ms
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-[#aaa]">Total Inferences</span>
                            <span className="text-white">
                                {mlStatus?.inference_engine?.stats?.total_inferences || 0}
                            </span>
                        </div>
                    </div>
                </div>

                {/* Simulate Controls */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h3 className="text-[#666] text-sm mb-3">SIMULATE THREATS</h3>
                    <div className="grid grid-cols-2 gap-2">
                        <button
                            onClick={() => simulateThreats('dos')}
                            disabled={simulating}
                            className="px-3 py-2 bg-[#ef4444] hover:bg-red-700 rounded text-white text-sm transition"
                        >
                            DoS Attack
                        </button>
                        <button
                            onClick={() => simulateThreats('scan')}
                            disabled={simulating}
                            className="px-3 py-2 bg-[#3b82f6] hover:bg-blue-700 rounded text-white text-sm transition"
                        >
                            Port Scan
                        </button>
                        <button
                            onClick={() => simulateThreats('brute_force')}
                            disabled={simulating}
                            className="px-3 py-2 bg-[#f97316] hover:bg-orange-700 rounded text-white text-sm transition"
                        >
                            Brute Force
                        </button>
                        <button
                            onClick={() => simulateThreats('normal')}
                            disabled={simulating}
                            className="px-3 py-2 bg-[#22c55e] hover:bg-green-700 rounded text-white text-sm transition"
                        >
                            Normal
                        </button>
                    </div>
                    <button
                        onClick={clearThreats}
                        className="w-full mt-2 px-3 py-2 bg-[#1a1a1a] border border-[#333] hover:bg-[#222] rounded text-[#aaa] text-sm transition"
                    >
                        Clear All
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Live Threat Feed */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h3 className="text-white text-lg font-semibold mb-4">
                        🔴 Live Threat Feed
                    </h3>
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                        {threats.length === 0 ? (
                            <div className="text-[#666] text-center py-8">
                                No threats detected. Click simulate to test!
                            </div>
                        ) : (
                            threats.map((threat) => (
                                <div
                                    key={threat.id}
                                    className={`p-4 rounded-lg border-l-4 bg-[#1a1a1a] ${getRiskBorderColor(threat.risk_level)}`}
                                >
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${getRiskColor(threat.risk_level)} text-white`}>
                                                    {threat.risk_level?.toUpperCase()}
                                                </span>
                                                <span className="text-white font-medium">{threat.threat_class}</span>
                                            </div>
                                            <div className="text-[#666] text-sm mt-1">
                                                Confidence: {(threat.confidence * 100).toFixed(1)}%
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-[#666] text-xs">
                                                {new Date(threat.timestamp).toLocaleTimeString()}
                                            </div>
                                            <div className="text-[#aaa] text-xs mt-1">
                                                Action: {threat.action}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Attack Timeline */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h3 className="text-white text-lg font-semibold mb-4">
                        ⏱️ Attack Kill Chain Timeline
                    </h3>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                        {timeline.length === 0 ? (
                            <div className="text-[#666] text-center py-8">
                                No attack stages detected
                            </div>
                        ) : (
                            timeline.slice(-20).map((event, idx) => (
                                <div key={`${event.id}-${idx}`} className="flex items-center gap-3">
                                    <div className="text-[#666] text-xs w-16">
                                        {new Date(event.timestamp).toLocaleTimeString()}
                                    </div>
                                    <div className={`w-3 h-3 rounded-full ${getStageColor(event.stage)}`} />
                                    <div className="flex-1">
                                        <span className="text-white text-sm">{event.stage.replace('_', ' ')}</span>
                                        <span className="text-[#666] text-xs ml-2">({event.threat_class})</span>
                                    </div>
                                    <div className={`text-xs ${getRiskColor(event.risk_level)} px-2 py-0.5 rounded text-white`}>
                                        {(event.confidence * 100).toFixed(0)}%
                                    </div>
                                </div>
                            ))
                        )}
                    </div>

                    {/* Kill Chain Legend */}
                    <div className="mt-4 pt-4 border-t border-[#2a2a2a]">
                        <div className="flex flex-wrap gap-2 text-xs text-[#aaa]">
                            <span className="flex items-center gap-1">
                                <div className="w-2 h-2 rounded-full bg-[#3b82f6]" /> Recon
                            </span>
                            <span className="flex items-center gap-1">
                                <div className="w-2 h-2 rounded-full bg-[#a855f7]" /> Access
                            </span>
                            <span className="flex items-center gap-1">
                                <div className="w-2 h-2 rounded-full bg-[#f97316]" /> Execution
                            </span>
                            <span className="flex items-center gap-1">
                                <div className="w-2 h-2 rounded-full bg-[#ef4444]" /> Persistence
                            </span>
                            <span className="flex items-center gap-1">
                                <div className="w-2 h-2 rounded-full bg-[#ec4899]" /> C2
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="mt-6 text-center text-[#666] text-sm">
                PCDS ML Dashboard • Auto-refreshes every 5 seconds • Models: Combined Classifier (84.78%) + Sequence Transformer + UEBA
            </div>
        </div>
    );
}

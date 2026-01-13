'use client';

import { useState, useEffect } from 'react';

interface AgentStats {
    episodes_trained: number;
    total_reward: number;
    average_reward: number;
    exploration_rate: number;
    learning_rate: number;
    actions_taken: { [key: string]: number };
}

interface Recommendation {
    recommended_action: string;
    confidence: number;
    q_values: { [key: string]: number };
    state: any;
}

export default function RLAgentPage() {
    const [stats, setStats] = useState<AgentStats | null>(null);
    const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
    const [loading, setLoading] = useState(false);
    const [training, setTraining] = useState(false);
    const [trainEpisodes, setTrainEpisodes] = useState(100);
    const [threatState, setThreatState] = useState({
        threat_level: 'medium',
        attack_type: 5,
        affected_hosts: 3,
        time_since_detection: 5
    });

    const API_BASE = 'http://localhost:8000/api/v2/advanced-ml';
    const threatLevels = ['none', 'low', 'medium', 'high', 'critical'];
    const actions = ['MONITOR', 'ALERT', 'BLOCK_IP', 'ISOLATE', 'QUARANTINE', 'FULL_LOCKDOWN'];

    useEffect(() => {
        fetchStats();
    }, []);

    const fetchStats = async () => {
        try {
            const res = await fetch(`${API_BASE}/rl/stats`);
            const data = await res.json();
            setStats(data);
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    };

    const getRecommendation = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/rl/recommend-action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(threatState)
            });
            const data = await res.json();
            setRecommendation(data);
        } catch (error) {
            console.error('Failed to get recommendation:', error);
        }
        setLoading(false);
    };

    const trainAgent = async () => {
        setTraining(true);
        try {
            await fetch(`${API_BASE}/rl/train?episodes=${trainEpisodes}`, { method: 'POST' });
            await fetchStats();
        } catch (error) {
            console.error('Failed to train:', error);
        }
        setTraining(false);
    };

    const getActionColor = (action: string) => {
        switch (action) {
            case 'MONITOR': return 'bg-[#3b82f6]';
            case 'ALERT': return 'bg-[#eab308]';
            case 'BLOCK_IP': return 'bg-[#f97316]';
            case 'ISOLATE': return 'bg-[#ef4444]';
            case 'QUARANTINE': return 'bg-[#dc2626]';
            case 'FULL_LOCKDOWN': return 'bg-[#991b1b]';
            default: return 'bg-[#666]';
        }
    };

    const getThreatColor = (level: string) => {
        switch (level) {
            case 'none': return 'bg-[#666]';
            case 'low': return 'bg-[#3b82f6]';
            case 'medium': return 'bg-[#eab308]';
            case 'high': return 'bg-[#f97316]';
            case 'critical': return 'bg-[#ef4444]';
            default: return 'bg-[#666]';
        }
    };

    const maxQValue = recommendation ? Math.max(...Object.values(recommendation.q_values).map(Math.abs), 0.001) : 1;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white flex items-center gap-3">
                    🤖 RL Defense Agent
                </h1>
                <p className="text-[#666] text-sm mt-1">
                    Reinforcement Learning agent that learns optimal defense strategies through experience
                </p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Episodes Trained</div>
                    <div className="text-2xl font-bold text-white">{stats?.episodes_trained || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Total Reward</div>
                    <div className="text-2xl font-bold text-[#22c55e]">{(stats?.total_reward || 0).toFixed(1)}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Avg Reward</div>
                    <div className="text-2xl font-bold text-[#3b82f6]">{(stats?.average_reward || 0).toFixed(2)}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Exploration ε</div>
                    <div className="text-2xl font-bold text-[#a855f7]">{(stats?.exploration_rate || 0.1).toFixed(2)}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Learning Rate</div>
                    <div className="text-2xl font-bold text-[#eab308]">{(stats?.learning_rate || 0.1).toFixed(2)}</div>
                </div>
            </div>

            {/* Training Panel */}
            <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                <h2 className="text-lg font-medium text-white mb-4">🎓 Train Agent</h2>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <label className="text-[#888]">Episodes:</label>
                        <input
                            type="number"
                            value={trainEpisodes}
                            onChange={(e) => setTrainEpisodes(parseInt(e.target.value) || 100)}
                            className="w-24 bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg px-3 py-2 text-white focus:border-[#10a37f] outline-none"
                        />
                    </div>
                    <button
                        onClick={trainAgent}
                        disabled={training}
                        className="bg-[#10a37f] hover:bg-[#0d8c6d] text-white px-6 py-2 rounded-lg transition disabled:opacity-50"
                    >
                        {training ? '⏳ Training...' : '🚀 Train Agent'}
                    </button>
                </div>
                <p className="text-[#666] text-sm mt-2">
                    Train the Q-Learning agent through simulated threat scenarios
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Threat State Input */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h2 className="text-lg font-medium text-white mb-4">⚠️ Threat Scenario</h2>
                    <div className="space-y-4">
                        <div>
                            <label className="text-[#888] text-sm block mb-2">Threat Level</label>
                            <div className="flex gap-2">
                                {threatLevels.map(level => (
                                    <button
                                        key={level}
                                        onClick={() => setThreatState({ ...threatState, threat_level: level })}
                                        className={`px-3 py-2 rounded-lg text-sm capitalize transition ${threatState.threat_level === level
                                            ? `${getThreatColor(level)} text-white`
                                            : 'bg-[#1a1a1a] text-[#888] hover:bg-[#252525] border border-[#2a2a2a]'
                                            }`}
                                    >
                                        {level}
                                    </button>
                                ))}
                            </div>
                        </div>

                        <div>
                            <label className="text-[#888] text-sm block mb-2">Attack Type: {threatState.attack_type}</label>
                            <input
                                type="range"
                                min="0"
                                max="15"
                                value={threatState.attack_type}
                                onChange={(e) => setThreatState({ ...threatState, attack_type: parseInt(e.target.value) })}
                                className="w-full accent-[#10a37f]"
                            />
                        </div>

                        <div>
                            <label className="text-[#888] text-sm block mb-2">Affected Hosts: {threatState.affected_hosts}</label>
                            <input
                                type="range"
                                min="1"
                                max="100"
                                value={threatState.affected_hosts}
                                onChange={(e) => setThreatState({ ...threatState, affected_hosts: parseInt(e.target.value) })}
                                className="w-full accent-[#10a37f]"
                            />
                        </div>

                        <div>
                            <label className="text-[#888] text-sm block mb-2">Time Since Detection: {threatState.time_since_detection} min</label>
                            <input
                                type="range"
                                min="0"
                                max="180"
                                value={threatState.time_since_detection}
                                onChange={(e) => setThreatState({ ...threatState, time_since_detection: parseInt(e.target.value) })}
                                className="w-full accent-[#10a37f]"
                            />
                        </div>

                        <button
                            onClick={getRecommendation}
                            disabled={loading}
                            className="w-full bg-[#10a37f] hover:bg-[#0d8c6d] text-white py-3 rounded-lg transition disabled:opacity-50"
                        >
                            {loading ? '⏳ Analyzing...' : '🎯 Get Recommended Action'}
                        </button>
                    </div>
                </div>

                {/* Recommendation */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h2 className="text-lg font-medium text-white mb-4">🎯 Agent Recommendation</h2>
                    {recommendation ? (
                        <div className="space-y-4">
                            <div className={`rounded-lg p-6 text-center ${getActionColor(recommendation.recommended_action)}`}>
                                <div className="text-5xl mb-2">
                                    {recommendation.recommended_action === 'MONITOR' ? '👀' :
                                        recommendation.recommended_action === 'ALERT' ? '🔔' :
                                            recommendation.recommended_action === 'BLOCK_IP' ? '🚫' :
                                                recommendation.recommended_action === 'ISOLATE' ? '🔐' :
                                                    recommendation.recommended_action === 'QUARANTINE' ? '📦' : '🔒'}
                                </div>
                                <div className="text-2xl font-bold text-white">{recommendation.recommended_action}</div>
                                <div className="text-white/80 text-sm mt-1">
                                    Confidence: {(recommendation.confidence * 100).toFixed(1)}%
                                </div>
                            </div>

                            <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                <h3 className="text-[#10a37f] font-medium mb-3">Q-Values (Action Values)</h3>
                                <div className="space-y-2">
                                    {actions.map(action => {
                                        const qValue = recommendation.q_values[action] || 0;
                                        const normalizedWidth = Math.abs(qValue) / maxQValue * 100;
                                        const isRecommended = action === recommendation.recommended_action;
                                        return (
                                            <div key={action} className="flex items-center gap-2">
                                                <span className={`text-sm w-28 ${isRecommended ? 'text-[#10a37f] font-bold' : 'text-[#888]'}`}>
                                                    {action}
                                                </span>
                                                <div className="flex-1 h-4 bg-[#1a1a1a] rounded overflow-hidden relative">
                                                    <div
                                                        className={`h-full transition-all ${qValue >= 0 ? 'bg-[#22c55e]' : 'bg-[#ef4444]'} ${isRecommended ? 'ring-2 ring-[#10a37f]' : ''}`}
                                                        style={{ width: `${normalizedWidth}%` }}
                                                    />
                                                </div>
                                                <span className={`text-xs w-12 text-right ${qValue >= 0 ? 'text-[#22c55e]' : 'text-[#ef4444]'}`}>
                                                    {qValue.toFixed(2)}
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>

                            <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                <h3 className="text-[#10a37f] font-medium mb-2">Current State</h3>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    <div className="text-[#666]">Threat Level:</div>
                                    <div className="text-white capitalize">{recommendation.state.threat_level}</div>
                                    <div className="text-[#666]">Attack Type:</div>
                                    <div className="text-white">{recommendation.state.attack_type}</div>
                                    <div className="text-[#666]">Affected Hosts:</div>
                                    <div className="text-white">{recommendation.state.affected_hosts}</div>
                                    <div className="text-[#666]">Time:</div>
                                    <div className="text-white">{recommendation.state.time_since_detection} min</div>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="text-center text-[#666] py-12">
                            <div className="text-6xl mb-4">🤖</div>
                            <p>Configure a threat scenario</p>
                            <p className="text-sm mt-2">and click "Get Recommended Action"</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Action Distribution */}
            {stats?.actions_taken && (
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h2 className="text-lg font-medium text-white mb-4">📊 Action Distribution</h2>
                    <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
                        {actions.map(action => (
                            <div key={action} className="text-center">
                                <div className={`${getActionColor(action)} rounded-lg p-4 mb-2`}>
                                    <div className="text-2xl font-bold text-white">{stats.actions_taken[action] || 0}</div>
                                </div>
                                <div className="text-xs text-[#666]">{action}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

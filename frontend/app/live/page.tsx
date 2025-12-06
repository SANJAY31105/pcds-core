'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { ThreatDetection, AlertNotification } from '@/types';
import { Activity, AlertTriangle, Zap } from 'lucide-react';

export default function LiveFeedPage() {
    const [threats, setThreats] = useState<ThreatDetection[]>([]);
    const [alerts, setAlerts] = useState<AlertNotification[]>([]);
    const [connected, setConnected] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Load initial data
        loadInitialData();
        setConnected(true); // Mark as connected (using polling instead of WebSocket)

        // Auto-refresh every 5 seconds for real-time updates
        const refreshInterval = setInterval(() => {
            loadInitialData();
        }, 5000);

        return () => {
            clearInterval(refreshInterval);
        };
    }, []);

    const loadInitialData = async () => {
        try {
            // Fetch recent detections (reduced limit for performance)
            const response = await apiClient.getDetections({ limit: 100, hours: 168 }) as any;

            // Map to ThreatDetection format
            const mappedDetections = (response.detections || []).map((d: any) => ({
                id: d.id || '',
                title: d.detection_type || 'Unknown Threat',
                description: d.description || '',
                severity: d.severity || 'medium',
                threat_type: d.detection_type || '',
                source_ip: d.source_ip || 'N/A',
                destination_ip: d.destination_ip || 'N/A',
                risk_score: d.risk_score || 0,
                timestamp: d.detected_at || new Date().toISOString(),
                mitre: d.technique_id ? {
                    technique_id: d.technique_id,
                    technique_name: d.technique_name || '',
                    tactic_id: d.tactic_id || '',
                    tactic_name: d.tactic_name || '',
                    kill_chain_stage: d.kill_chain_stage || 0,
                    severity: d.severity || ''
                } : undefined
            }));

            setThreats(mappedDetections);
            setAlerts([]); // Alerts API not implemented yet
        } catch (error) {
            console.error('Failed to load data:', error);
        } finally {
            setLoading(false);
        }
    };

    const getSeverityColor = (severity: string) => {
        const colors = {
            critical: 'bg-red-500/20 text-red-400 border-red-500/50',
            high: 'bg-orange-500/20 text-orange-400 border-orange-500/50',
            medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
            low: 'bg-blue-500/20 text-blue-400 border-blue-500/50'
        };
        return colors[severity as keyof typeof colors] || colors.low;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                        Live Threat Feed
                    </h1>
                    <p className="text-slate-400 mt-1">Real-time threat detection stream</p>
                </div>
                <div className={`px-4 py-2 rounded-lg flex items-center space-x-2 ${connected ? 'bg-green-500/20 border border-green-500/50' : 'bg-red-500/20 border border-red-500/50'}`}>
                    <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
                    <span className={`text-sm font-medium ${connected ? 'text-green-400' : 'text-red-400'}`}>
                        {connected ? 'Live' : 'Disconnected'}
                    </span>
                </div>
            </div>

            {/* Alerts Bar */}
            {alerts.length > 0 && (
                <div className="bg-gradient-to-r from-red-500/10 to-orange-500/10 rounded-xl border border-red-500/30 p-4">
                    <div className="flex items-center space-x-3">
                        <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0" />
                        <div className="flex-1">
                            <h3 className="font-medium text-white">Latest Alert</h3>
                            <p className="text-sm text-slate-300">{alerts[0].message}</p>
                        </div>
                        <span className={`px-3 py-1 text-xs font-bold rounded-full border ${getSeverityColor(alerts[0].severity)}`}>
                            {alerts[0].severity.toUpperCase()}
                        </span>
                    </div>
                </div>
            )}

            {/* Live Feed */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 shadow-2xl">
                <div className="p-6 border-b border-cyan-500/20">
                    <div className="flex items-center space-x-3">
                        <Activity className="w-6 h-6 text-cyan-400 animate-pulse" />
                        <h2 className="text-xl font-bold text-white">Real-Time Detections</h2>
                        <span className="px-3 py-1 bg-cyan-500/20 text-cyan-400 text-sm font-medium rounded-full">
                            {threats.length} events
                        </span>
                    </div>
                </div>

                <div className="max-h-[600px] overflow-y-auto">
                    {loading ? (
                        <div className="p-12 text-center">
                            <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                            <p className="text-slate-400">Loading threat detections...</p>
                        </div>
                    ) : threats.length === 0 ? (
                        <div className="p-12 text-center text-slate-400">
                            <Activity className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                            <p className="text-lg font-medium mb-2">No Detections Yet</p>
                            <p className="text-sm">The system is monitoring for threats. Recent detections will appear here in real-time.</p>
                        </div>
                    ) : (
                        <div className="divide-y divide-slate-700/50">
                            {threats.map((threat, index) => (
                                <div
                                    key={threat.id}
                                    className="p-6 hover:bg-slate-800/30 transition-colors animate-fadeIn"
                                    style={{ animationDelay: `${index * 0.05}s` }}
                                >
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-3 mb-2">
                                                <Zap className="w-5 h-5 text-yellow-400" />
                                                <h4 className="font-medium text-white">{threat.title}</h4>
                                                <span className={`px-3 py-1 text-xs font-bold rounded-full border ${getSeverityColor(threat.severity)}`}>
                                                    {threat.severity.toUpperCase()}
                                                </span>
                                            </div>
                                            <p className="text-sm text-slate-400 mb-3">{threat.description}</p>
                                            <div className="flex items-center space-x-6 text-xs text-slate-500">
                                                <span className="flex items-center">
                                                    <span className="mr-1">Source:</span>
                                                    <span className="font-mono text-cyan-400">{threat.source_ip}</span>
                                                </span>
                                                <span>Risk: {threat.risk_score}</span>
                                                <span>{new Date(threat.timestamp).toLocaleTimeString()}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <style jsx>{`
                @keyframes fadeIn {
                    from {
                        opacity: 0;
                        transform: translateY(-10px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                .animate-fadeIn {
                    animation: fadeIn 0.3s ease-out forwards;
                }
            `}</style>
        </div>
    );
}

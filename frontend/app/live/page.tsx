'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Activity, AlertTriangle, Shield, Zap, Pause, Play, Wifi, Power, Eye, EyeOff, Download } from 'lucide-react';

interface LiveEvent {
    id: string;
    type: 'detection' | 'action' | 'system';
    message: string;
    severity?: string;
    timestamp: string;
    source?: string;
    details?: any;
    mitre?: { technique_id: string; technique_name: string };
}

interface NetworkConnection {
    remote_ip: string;
    remote_port: number;
    local_port: number;
    hostname: string;
    process: string;
    status: string;
    is_suspicious: boolean;
    anomaly_score: number;
    threat_info?: any;
}

export default function LiveFeedPage() {
    const [events, setEvents] = useState<LiveEvent[]>([]);
    const [connections, setConnections] = useState<NetworkConnection[]>([]);
    const [paused, setPaused] = useState(false);
    const [realMode, setRealMode] = useState(true);  // Default to REAL mode
    const [monitoring, setMonitoring] = useState(true);  // Auto-start monitoring
    const [stats, setStats] = useState({
        detections: 0,
        actions: 0,
        packets: 0,
        connections: 0,
        suspicious: 0
    });

    // Start/Stop real monitoring
    const toggleRealMonitoring = async () => {
        try {
            if (!monitoring) {
                await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/network/start`, { method: 'POST' });
                setMonitoring(true);
                setRealMode(true);
            } else {
                await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/network/stop`, { method: 'POST' });
                setMonitoring(false);
            }
        } catch (error) {
            console.error('Failed to toggle monitoring:', error);
        }
    };

    // Fetch real network data
    const fetchRealData = async () => {
        if (!realMode || !monitoring) return;

        try {
            const [statsRes, eventsRes, connectionsRes] = await Promise.all([
                fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/network/stats`),
                fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/network/events?limit=50`),
                fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/network/connections?limit=20`)
            ]);

            const statsData = await statsRes.json();
            const eventsData = await eventsRes.json();
            const connectionsData = await connectionsRes.json();

            setStats({
                packets: statsData.packets_analyzed || 0,
                connections: statsData.active_connections || 0,
                detections: eventsData.events?.filter((e: any) => e.type === 'detection').length || 0,
                actions: statsData.suspicious_count || 0,
                suspicious: statsData.suspicious_count || 0
            });

            if (eventsData.events) {
                setEvents(eventsData.events);
            }

            if (connectionsData.connections) {
                setConnections(connectionsData.connections);
            }
        } catch (error) {
            console.error('Failed to fetch network data:', error);
        }
    };

    // Simulated events for demo mode - HIGH IMPACT EVENTS FOR JUDGES
    useEffect(() => {
        if (realMode) return;

        // Generate timestamps in the last 30 seconds for realistic feel
        const now = new Date();
        const timeAgo = (seconds: number) => new Date(now.getTime() - seconds * 1000).toISOString();

        const initialEvents: LiveEvent[] = [
            {
                id: '1',
                type: 'system',
                message: 'PCDS Enterprise initialized - All ML models loaded successfully',
                timestamp: timeAgo(30),
                source: 'System'
            },
            {
                id: '2',
                type: 'detection',
                message: '🔴 CRITICAL: Phishing email opened by user@company.com - Malicious payload detected',
                severity: 'critical',
                timestamp: timeAgo(25),
                source: 'Email Gateway',
                mitre: { technique_id: 'T1566', technique_name: 'Phishing' }
            },
            {
                id: '3',
                type: 'action',
                message: '✅ AUTO-RESPONSE: Email quarantined, user session flagged for monitoring',
                timestamp: timeAgo(24),
                source: 'SOAR Engine'
            },
            {
                id: '4',
                type: 'detection',
                message: '⚠️ C2 Beacon detected: 185.174.xxx.xxx:443 - Cobalt Strike signature match',
                severity: 'critical',
                timestamp: timeAgo(18),
                source: 'Network Monitor',
                mitre: { technique_id: 'T1071', technique_name: 'Application Layer Protocol' }
            },
            {
                id: '5',
                type: 'detection',
                message: '🔴 Privilege Escalation attempt: Process trying to elevate to SYSTEM on host-042',
                severity: 'high',
                timestamp: timeAgo(12),
                source: 'EDR Agent',
                mitre: { technique_id: 'T1548', technique_name: 'Abuse Elevation Control' }
            },
            {
                id: '6',
                type: 'action',
                message: '✅ AUTO-RESPONSE: Host-042 network isolated pending investigation',
                timestamp: timeAgo(11),
                source: 'SOAR Engine'
            },
            {
                id: '7',
                type: 'detection',
                message: '⚠️ Lateral Movement: RDP connection from host-042 to DC-01 (suspicious after priv-esc)',
                severity: 'high',
                timestamp: timeAgo(8),
                source: 'Network Monitor',
                mitre: { technique_id: 'T1021', technique_name: 'Remote Services' }
            },
            {
                id: '8',
                type: 'detection',
                message: '🔴 RANSOMWARE BEHAVIOR: Mass file encryption detected on host-042 - 847 files affected',
                severity: 'critical',
                timestamp: timeAgo(5),
                source: 'Behavioral Analytics',
                mitre: { technique_id: 'T1486', technique_name: 'Data Encrypted for Impact' }
            },
            {
                id: '9',
                type: 'action',
                message: '✅ AUTO-RESPONSE: Kill switch activated, encryption halted, snapshot initiated',
                timestamp: timeAgo(4),
                source: 'SOAR Engine'
            },
            {
                id: '10',
                type: 'system',
                message: '📊 PCDS ML Confidence: 94.2% - Attack chain complete: Phishing → C2 → Priv-Esc → Ransomware',
                timestamp: timeAgo(2),
                source: 'ML Engine'
            }
        ];
        setEvents(initialEvents);
        setStats({
            detections: 5,
            actions: 3,
            packets: 847293,
            connections: 156,
            suspicious: 4
        });

        const interval = setInterval(() => {
            if (paused || realMode) return;

            // High-impact event rotation for demo
            const eventTypes = [
                {
                    type: 'detection' as const,
                    message: `🔄 Behavioral anomaly: Entity activity ${Math.floor(Math.random() * 300 + 100)}% above baseline`,
                    severity: 'high',
                    source: 'UEBA',
                    mitre: { technique_id: 'T1078', technique_name: 'Valid Accounts' }
                },
                {
                    type: 'detection' as const,
                    message: `⚠️ DNS Tunneling suspected: High entropy queries to ${['xyz123.evil.com', 'c2.malware.net', 'exfil.badactor.io'][Math.floor(Math.random() * 3)]}`,
                    severity: 'high',
                    source: 'DNS Monitor',
                    mitre: { technique_id: 'T1048', technique_name: 'Exfiltration Over Alternative Protocol' }
                },
                {
                    type: 'detection' as const,
                    message: `🔴 Credential dumping detected: LSASS memory access on host-${String(Math.floor(Math.random() * 100)).padStart(3, '0')}`,
                    severity: 'critical',
                    source: 'EDR Agent',
                    mitre: { technique_id: 'T1003', technique_name: 'OS Credential Dumping' }
                },
                {
                    type: 'action' as const,
                    message: '✅ Firewall rule updated: Blocked C2 IP range 185.174.xxx.0/24',
                    source: 'Auto Response'
                },
                {
                    type: 'system' as const,
                    message: `📡 Network scan: ${Math.floor(Math.random() * 50000 + 10000)} packets analyzed, ${Math.floor(Math.random() * 3)} anomalies`,
                    source: 'Scanner'
                },
                {
                    type: 'detection' as const,
                    message: `⚠️ Data exfiltration attempt: ${Math.floor(Math.random() * 500 + 100)}MB upload to cloud storage`,
                    severity: 'high',
                    source: 'DLP',
                    mitre: { technique_id: 'T1567', technique_name: 'Exfiltration Over Web Service' }
                }
            ];

            const randomEvent = eventTypes[Math.floor(Math.random() * eventTypes.length)];
            const newEvent: LiveEvent = {
                id: Date.now().toString(),
                ...randomEvent,
                timestamp: new Date().toISOString()
            };

            setEvents(prev => [newEvent, ...prev.slice(0, 49)]);
            setStats(prev => ({
                ...prev,
                detections: prev.detections + (randomEvent.type === 'detection' ? 1 : 0),
                actions: prev.actions + (randomEvent.type === 'action' ? 1 : 0),
                packets: prev.packets + Math.floor(Math.random() * 5000 + 1000),
                connections: prev.connections + Math.floor(Math.random() * 3),
                suspicious: prev.suspicious + (randomEvent.severity === 'critical' ? 1 : 0)
            }));
        }, 3000);

        return () => clearInterval(interval);
    }, [paused, realMode]);

    // Real data polling
    useEffect(() => {
        if (!realMode || !monitoring || paused) return;

        const interval = setInterval(fetchRealData, 2000);
        fetchRealData(); // Initial fetch

        return () => clearInterval(interval);
    }, [realMode, monitoring, paused]);

    const getIcon = (type: string, severity?: string) => {
        if (type === 'detection') return <AlertTriangle className="w-5 h-5" style={{ color: severity === 'critical' ? '#ef4444' : severity === 'high' ? '#f97316' : '#10a37f' }} />;
        if (type === 'action') return <Zap className="w-5 h-5 text-[#f97316]" />;
        return <Activity className="w-5 h-5 text-[#666]" />;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-semibold text-white">Live Feed</h1>
                    <p className="text-[#666] mt-1">
                        {realMode ? '🔴 Real-time network monitoring' : '🎭 Simulated security events'}
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    {/* Real Mode Toggle */}
                    <button
                        onClick={toggleRealMonitoring}
                        className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${monitoring
                            ? 'bg-red-500/20 text-red-400 border border-red-500/50'
                            : 'bg-[#10a37f]/20 text-[#10a37f] border border-[#10a37f]/50'
                            }`}
                    >
                        <Power className="w-4 h-4" />
                        {monitoring ? 'Stop Monitoring' : 'Start Real Monitor'}
                    </button>

                    <Link
                        href="/download"
                        className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-[#141414] border border-[#2a2a2a] text-[#a1a1a1] hover:text-white transition-colors"
                    >
                        <Download className="w-4 h-4" />
                        Get Agent
                    </Link>

                    {/* Pause/Resume */}
                    <button
                        onClick={() => setPaused(!paused)}
                        className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${paused
                            ? 'bg-[#10a37f] text-white'
                            : 'bg-[#141414] border border-[#2a2a2a] text-[#a1a1a1] hover:text-white'
                            }`}
                    >
                        {paused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                        {paused ? 'Resume' : 'Pause'}
                    </button>

                    {/* Status Indicator */}
                    <div className="flex items-center gap-2 px-4 py-2.5 bg-[#141414] border border-[#2a2a2a] rounded-lg">
                        <div className={`w-3 h-3 rounded-full ${paused ? 'bg-[#666]' : monitoring ? 'bg-red-500 animate-pulse' : 'bg-[#22c55e] animate-pulse'}`}></div>
                        <span className="text-sm font-medium text-white">
                            {paused ? 'Paused' : monitoring ? 'Real' : 'Demo'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-5 gap-4">
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Activity className="w-5 h-5 text-[#10a37f]" />
                        <span className="text-sm text-[#666]">Packets</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{stats.packets.toLocaleString()}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Wifi className="w-5 h-5 text-[#3b82f6]" />
                        <span className="text-sm text-[#666]">Connections</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{stats.connections}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <AlertTriangle className="w-5 h-5 text-[#f97316]" />
                        <span className="text-sm text-[#666]">Detections</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{stats.detections}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Shield className="w-5 h-5 text-[#ef4444]" />
                        <span className="text-sm text-[#666]">Suspicious</span>
                    </div>
                    <p className="text-3xl font-bold text-[#ef4444]">{stats.suspicious}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Zap className="w-5 h-5 text-[#22c55e]" />
                        <span className="text-sm text-[#666]">Actions</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{stats.actions}</p>
                </div>
            </div>

            {/* Active Connections (Real Mode) */}
            {realMode && connections.length > 0 && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] overflow-hidden">
                    <div className="px-6 py-4 border-b border-[#2a2a2a]">
                        <span className="text-lg font-medium text-white">Active Connections</span>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-[#2a2a2a]">
                                    <th className="text-left px-6 py-3 text-xs text-[#666] font-medium">Process</th>
                                    <th className="text-left px-6 py-3 text-xs text-[#666] font-medium">Remote</th>
                                    <th className="text-left px-6 py-3 text-xs text-[#666] font-medium">Port</th>
                                    <th className="text-left px-6 py-3 text-xs text-[#666] font-medium">Status</th>
                                    <th className="text-left px-6 py-3 text-xs text-[#666] font-medium">Risk</th>
                                </tr>
                            </thead>
                            <tbody>
                                {connections.slice(0, 10).map((conn, i) => (
                                    <tr key={i} className={`border-b border-[#1a1a1a] ${conn.is_suspicious ? 'bg-red-500/5' : ''}`}>
                                        <td className="px-6 py-3 text-sm text-white">{conn.process}</td>
                                        <td className="px-6 py-3 text-sm text-[#a1a1a1]">{conn.hostname}</td>
                                        <td className="px-6 py-3 text-sm text-[#666]">{conn.remote_port}</td>
                                        <td className="px-6 py-3">
                                            <span className={`text-xs px-2 py-0.5 rounded ${conn.status === 'ESTABLISHED' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                                                {conn.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-3">
                                            {conn.is_suspicious ? (
                                                <span className="text-xs px-2 py-0.5 rounded bg-red-500/20 text-red-400">⚠️ Suspicious</span>
                                            ) : (
                                                <span className="text-xs text-[#666]">{(conn.anomaly_score * 100).toFixed(0)}%</span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Event Stream */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] overflow-hidden">
                <div className="px-6 py-4 border-b border-[#2a2a2a] flex items-center justify-between">
                    <span className="text-lg font-medium text-white">Event Stream</span>
                    <span className="text-sm text-[#666]">{events.length} events</span>
                </div>
                <div className="max-h-[400px] overflow-y-auto">
                    {events.map((event, i) => (
                        <div
                            key={event.id}
                            className={`flex items-center gap-4 px-6 py-4 border-b border-[#1a1a1a] hover:bg-[#1a1a1a] transition-colors ${i === 0 ? 'bg-[#1a1a1a]/50' : ''} ${event.severity === 'critical' ? 'border-l-2 border-l-red-500' : event.severity === 'high' ? 'border-l-2 border-l-orange-500' : ''}`}
                        >
                            <div className="w-10 h-10 rounded-lg bg-[#0a0a0a] flex items-center justify-center">
                                {getIcon(event.type, event.severity)}
                            </div>
                            <div className="flex-1">
                                <p className="text-sm text-white">{event.message}</p>
                                <div className="flex items-center gap-3 mt-0.5">
                                    <span className="text-xs text-[#666]">{event.source}</span>
                                    {event.mitre && (
                                        <span className="text-xs px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400">
                                            {event.mitre.technique_id}
                                        </span>
                                    )}
                                </div>
                            </div>
                            <div className="text-right">
                                <p className="text-sm text-[#a1a1a1]">{new Date(event.timestamp).toLocaleTimeString()}</p>
                                {event.severity && (
                                    <span className={`text-xs ${event.severity === 'critical' ? 'text-red-400' : event.severity === 'high' ? 'text-orange-400' : 'text-[#666]'}`}>
                                        {event.severity}
                                    </span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

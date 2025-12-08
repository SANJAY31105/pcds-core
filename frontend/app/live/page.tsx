'use client';

import { useState, useEffect, useRef } from 'react';
import { Activity, AlertTriangle, Shield, Zap, Clock, Pause, Play, Wifi } from 'lucide-react';

interface LiveEvent {
    id: string;
    type: 'detection' | 'action' | 'system';
    message: string;
    severity?: string;
    timestamp: string;
    source?: string;
}

export default function LiveFeedPage() {
    const [events, setEvents] = useState<LiveEvent[]>([]);
    const [paused, setPaused] = useState(false);
    const [stats, setStats] = useState({ detections: 0, actions: 0, packets: 0, connections: 0 });

    useEffect(() => {
        const initialEvents: LiveEvent[] = [
            { id: '1', type: 'system', message: 'PCDS Enterprise initialized - All systems operational', timestamp: new Date().toISOString(), source: 'System' },
            { id: '2', type: 'system', message: 'ML anomaly detection models loaded successfully', timestamp: new Date().toISOString(), source: 'ML Engine' },
            { id: '3', type: 'detection', message: 'Network baseline established - monitoring 156 endpoints', severity: 'low', timestamp: new Date().toISOString(), source: 'Network Monitor' }
        ];
        setEvents(initialEvents);

        const interval = setInterval(() => {
            if (paused) return;

            const eventTypes = [
                { type: 'system' as const, message: `Network scan completed: ${Math.floor(Math.random() * 5000 + 1000)} packets analyzed`, source: 'Scanner' },
                { type: 'system' as const, message: `Endpoint health check: ${Math.floor(Math.random() * 50 + 100)} hosts responding`, source: 'Health Monitor' },
                { type: 'system' as const, message: `DNS query logged: ${['google.com', 'microsoft.com', 'github.com', 'aws.amazon.com'][Math.floor(Math.random() * 4)]}`, source: 'DNS Monitor' },
                { type: 'detection' as const, message: 'New connection established from internal network', severity: 'low', source: 'Firewall' },
                { type: 'action' as const, message: 'Firewall rule updated - blocked suspicious IP range', source: 'Auto Response' }
            ];

            const randomEvent = eventTypes[Math.floor(Math.random() * eventTypes.length)];
            const newEvent: LiveEvent = {
                id: Date.now().toString(),
                ...randomEvent,
                timestamp: new Date().toISOString()
            };

            setEvents(prev => [newEvent, ...prev.slice(0, 99)]);
            setStats(prev => ({
                detections: prev.detections + (randomEvent.type === 'detection' ? 1 : 0),
                actions: prev.actions + (randomEvent.type === 'action' ? 1 : 0),
                packets: prev.packets + Math.floor(Math.random() * 2000 + 500),
                connections: prev.connections + Math.floor(Math.random() * 5)
            }));
        }, 1500);

        return () => clearInterval(interval);
    }, [paused]);

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
                    <p className="text-[#666] mt-1">Real-time security event monitoring</p>
                </div>
                <div className="flex items-center gap-4">
                    <button
                        onClick={() => setPaused(!paused)}
                        className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-colors ${paused
                                ? 'bg-[#10a37f] text-white'
                                : 'bg-[#141414] border border-[#2a2a2a] text-[#a1a1a1] hover:text-white'
                            }`}
                    >
                        {paused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                        {paused ? 'Resume' : 'Pause'}
                    </button>
                    <div className="flex items-center gap-2 px-4 py-2.5 bg-[#141414] border border-[#2a2a2a] rounded-lg">
                        <div className={`w-3 h-3 rounded-full ${paused ? 'bg-[#666]' : 'bg-[#22c55e] animate-pulse'}`}></div>
                        <span className="text-sm font-medium text-white">{paused ? 'Paused' : 'Live'}</span>
                    </div>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-4 gap-4">
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                    <div className="flex items-center gap-3 mb-3">
                        <Activity className="w-6 h-6 text-[#10a37f]" />
                        <span className="text-sm text-[#666]">Packets Analyzed</span>
                    </div>
                    <p className="text-4xl font-bold text-white">{stats.packets.toLocaleString()}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                    <div className="flex items-center gap-3 mb-3">
                        <Wifi className="w-6 h-6 text-[#3b82f6]" />
                        <span className="text-sm text-[#666]">Active Connections</span>
                    </div>
                    <p className="text-4xl font-bold text-white">{stats.connections}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                    <div className="flex items-center gap-3 mb-3">
                        <AlertTriangle className="w-6 h-6 text-[#f97316]" />
                        <span className="text-sm text-[#666]">Detections</span>
                    </div>
                    <p className="text-4xl font-bold text-white">{stats.detections}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                    <div className="flex items-center gap-3 mb-3">
                        <Zap className="w-6 h-6 text-[#22c55e]" />
                        <span className="text-sm text-[#666]">Actions Taken</span>
                    </div>
                    <p className="text-4xl font-bold text-white">{stats.actions}</p>
                </div>
            </div>

            {/* Event Stream */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] overflow-hidden">
                <div className="px-6 py-4 border-b border-[#2a2a2a] flex items-center justify-between">
                    <span className="text-lg font-medium text-white">Event Stream</span>
                    <span className="text-sm text-[#666]">{events.length} events captured</span>
                </div>
                <div className="max-h-[500px] overflow-y-auto">
                    {events.map((event, i) => (
                        <div
                            key={event.id}
                            className={`flex items-center gap-4 px-6 py-4 border-b border-[#1a1a1a] hover:bg-[#1a1a1a] transition-colors ${i === 0 ? 'bg-[#1a1a1a]/50' : ''}`}
                        >
                            <div className="w-10 h-10 rounded-lg bg-[#0a0a0a] flex items-center justify-center">
                                {getIcon(event.type, event.severity)}
                            </div>
                            <div className="flex-1">
                                <p className="text-sm text-white">{event.message}</p>
                                <p className="text-xs text-[#666] mt-0.5">{event.source}</p>
                            </div>
                            <div className="text-right">
                                <p className="text-sm text-[#a1a1a1]">{new Date(event.timestamp).toLocaleTimeString()}</p>
                                {event.severity && (
                                    <span className="text-xs text-[#666]">{event.severity}</span>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

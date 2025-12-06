'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { ThreatDetection } from '@/types';
import { AlertTriangle, Filter, Clock, Target, Search } from 'lucide-react';

export default function DetectionsPage() {
    const [detections, setDetections] = useState<ThreatDetection[]>([]);
    const [allDetections, setAllDetections] = useState<ThreatDetection[]>([]);
    const [loading, setLoading] = useState(true);
    const [severityFilter, setSeverityFilter] = useState('');
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        loadDetections();
    }, [severityFilter]);

    const loadDetections = async () => {
        try {
            const params: any = { limit: 20, hours: 24 }; // Optimized: 20 items, last 24h
            if (severityFilter) params.severity = severityFilter;

            const response = await apiClient.getDetections(params) as any;

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

            setAllDetections(mappedDetections);
            setDetections(mappedDetections);
        } catch (error) {
            console.error('Failed to load detections:', error);
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

    const handleSearch = (query: string) => {
        setSearchQuery(query);
        if (query.trim()) {
            const filtered = allDetections.filter(d =>
                d.title.toLowerCase().includes(query.toLowerCase()) ||
                d.description.toLowerCase().includes(query.toLowerCase()) ||
                d.source_ip.toLowerCase().includes(query.toLowerCase()) ||
                d.destination_ip.toLowerCase().includes(query.toLowerCase())
            );
            setDetections(filtered);
        } else {
            setDetections(allDetections);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                        Detection Feed
                    </h1>
                    <p className="text-slate-400 mt-1">Real-time threat detections with MITRE ATT&CK mapping</p>
                </div>
            </div>

            {/* Search and Filters */}
            <div className="space-y-4">
                {/* Search Box */}
                <div className="relative">
                    <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
                    <input
                        type="text"
                        placeholder="Search detections by type, IP, description..."
                        value={searchQuery}
                        onChange={(e) => handleSearch(e.target.value)}
                        className="w-full pl-12 pr-4 py-3 bg-slate-800/50 border border-cyan-500/20 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20"
                    />
                </div>

                {/* Severity Filters */}
                <div className="flex items-center space-x-4">
                    {['', 'critical', 'high', 'medium', 'low'].map((severity) => (
                        <button
                            key={severity || 'all'}
                            onClick={() => setSeverityFilter(severity)}
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${severityFilter === severity
                                ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/50'
                                : 'bg-slate-800/50 text-slate-400 hover:text-white border border-cyan-500/20'
                                }`}
                        >
                            {severity || 'All'}
                        </button>
                    ))}
                </div>

                {/* Results Info */}
                <div className="flex items-center justify-between text-sm text-slate-400">
                    <div>
                        Showing <span className="text-white font-bold">{detections.length}</span> of <span className="text-white font-bold">{allDetections.length}</span> detections
                    </div>
                    {searchQuery && (
                        <button
                            onClick={() => handleSearch('')}
                            className="text-cyan-400 hover:text-cyan-300 transition-colors"
                        >
                            Clear search
                        </button>
                    )}
                </div>
            </div>

            {/* Detections List */}
            <div className="space-y-4">
                {loading ? (
                    <div className="text-center py-12 text-slate-400">
                        Loading detections...
                    </div>
                ) : detections.length === 0 ? (
                    <div className="text-center py-12 text-slate-400">
                        No detections found
                    </div>
                ) : (
                    detections.map((detection) => (
                        <div
                            key={detection.id}
                            className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 p-6 shadow-xl hover:shadow-2xl hover:border-cyan-500/40 transition-all"
                        >
                            <div className="flex items-start justify-between">
                                <div className="flex-1">
                                    <div className="flex items-center space-x-3 mb-3">
                                        <h3 className="text-lg font-bold text-white">{detection.title}</h3>
                                        <span className={`px-3 py-1 text-xs font-bold rounded-full border ${getSeverityColor(detection.severity)}`}>
                                            {detection.severity.toUpperCase()}
                                        </span>
                                    </div>
                                    <p className="text-slate-400 mb-4">{detection.description}</p>

                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                                        <div>
                                            <p className="text-xs text-slate-500 mb-1">Source IP</p>
                                            <p className="text-sm font-mono text-cyan-400">{detection.source_ip}</p>
                                        </div>
                                        <div>
                                            <p className="text-xs text-slate-500 mb-1">Destination IP</p>
                                            <p className="text-sm font-mono text-cyan-400">{detection.destination_ip}</p>
                                        </div>
                                        <div>
                                            <p className="text-xs text-slate-500 mb-1">Risk Score</p>
                                            <p className="text-sm font-bold text-white">{detection.risk_score}/100</p>
                                        </div>
                                        <div>
                                            <p className="text-xs text-slate-500 mb-1">Detected</p>
                                            <p className="text-sm text-slate-300">{new Date(detection.timestamp).toLocaleString()}</p>
                                        </div>
                                    </div>

                                    {detection.mitre && (
                                        <div className="flex items-center space-x-3">
                                            <span className="px-3 py-1 text-xs font-mono bg-slate-900/50 text-cyan-400 rounded border border-cyan-500/30">
                                                {detection.mitre.technique_id}
                                            </span>
                                            <span className="text-sm text-slate-300">{detection.mitre.technique_name}</span>
                                            <span className="text-xs text-slate-500">•</span>
                                            <span className="text-sm text-slate-400">{detection.mitre.tactic_name}</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

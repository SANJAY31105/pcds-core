'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { ThreatDetection } from '@/types';
import { AlertTriangle, Search, ChevronRight } from 'lucide-react';

export default function DetectionsPage() {
    const [detections, setDetections] = useState<ThreatDetection[]>([]);
    const [loading, setLoading] = useState(true);
    const [severityFilter, setSeverityFilter] = useState('');
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        loadDetections();
    }, [severityFilter]);

    const loadDetections = async () => {
        try {
            const params: any = { limit: 20, hours: 24 };
            if (severityFilter) params.severity = severityFilter;
            const response = await apiClient.getDetections(params) as any;

            const mapped = (response.detections || []).map((d: any) => ({
                id: d.id || '',
                title: d.detection_type || 'Unknown Threat',
                description: d.description || '',
                severity: d.severity || 'medium',
                threat_type: d.detection_type || '',
                source_ip: d.source_ip || 'N/A',
                destination_ip: d.destination_ip || 'N/A',
                risk_score: d.risk_score || 0,
                timestamp: d.detected_at || new Date().toISOString(),
                mitre: d.technique_id ? { technique_id: d.technique_id, technique_name: d.technique_name || '' } : undefined
            }));
            setDetections(mapped);
        } catch (error) {
            console.error('Failed to load detections:', error);
        } finally {
            setLoading(false);
        }
    };

    const getSeverityColor = (severity: string) => {
        const colors: Record<string, string> = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#3b82f6' };
        return colors[severity] || colors.medium;
    };

    const filteredDetections = detections.filter(d =>
        d.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        d.source_ip.includes(searchQuery)
    );

    return (
        <div className="min-h-screen bg-[#0a0a0a] p-6 space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white">Detections</h1>
                <p className="text-[#666] text-sm mt-1">Real-time threat detections (Last 24h)</p>
            </div>

            {/* Filters */}
            <div className="flex gap-3">
                <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                    <input
                        type="text"
                        placeholder="Search detections..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 bg-[#141414] border border-[#2a2a2a] rounded-lg text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f] text-sm"
                    />
                </div>
                <div className="flex gap-1 p-1 bg-[#141414] rounded-lg border border-[#2a2a2a]">
                    {['', 'critical', 'high', 'medium', 'low'].map((sev) => (
                        <button
                            key={sev}
                            onClick={() => setSeverityFilter(sev)}
                            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${severityFilter === sev ? 'bg-[#1a1a1a] text-white' : 'text-[#666] hover:text-[#a1a1a1]'
                                }`}
                        >
                            {sev ? sev.charAt(0).toUpperCase() + sev.slice(1) : 'All'}
                        </button>
                    ))}
                </div>
            </div>

            {/* Detections List */}
            <div className="space-y-2">
                {loading ? (
                    <div className="text-center py-12 text-[#666]">Loading...</div>
                ) : filteredDetections.length === 0 ? (
                    <div className="text-center py-12">
                        <AlertTriangle className="w-10 h-10 mx-auto mb-2 text-[#333]" />
                        <p className="text-[#666]">No detections found</p>
                    </div>
                ) : (
                    filteredDetections.map((detection) => (
                        <div key={detection.id} className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-4 hover:bg-[#1a1a1a] transition-colors cursor-pointer group">
                            <div className="flex items-start justify-between">
                                <div className="flex items-start gap-4">
                                    <div className="w-2 h-2 rounded-full mt-2" style={{ backgroundColor: getSeverityColor(detection.severity) }}></div>
                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <h3 className="text-sm font-medium text-white group-hover:text-[#10a37f] transition-colors">{detection.title}</h3>
                                            <span className="text-xs px-2 py-0.5 rounded font-mono" style={{ backgroundColor: `${getSeverityColor(detection.severity)}20`, color: getSeverityColor(detection.severity) }}>
                                                {detection.severity.toUpperCase()}
                                            </span>
                                        </div>
                                        <p className="text-xs text-[#666] mb-2">{detection.description?.substring(0, 100)}...</p>
                                        <div className="flex items-center gap-4 text-xs text-[#666]">
                                            <span>Source: <span className="text-[#a1a1a1]">{detection.source_ip}</span></span>
                                            <span>→</span>
                                            <span>Dest: <span className="text-[#a1a1a1]">{detection.destination_ip}</span></span>
                                            {detection.mitre && (
                                                <span className="text-[#10a37f]">{detection.mitre.technique_id}</span>
                                            )}
                                        </div>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs text-[#666]">{new Date(detection.timestamp).toLocaleTimeString()}</p>
                                    <ChevronRight className="w-4 h-4 text-[#444] mt-2 ml-auto group-hover:text-[#10a37f]" />
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

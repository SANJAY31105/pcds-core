'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { apiClient } from '@/lib/api';
import { Entity, Detection } from '@/types';
import Link from 'next/link';
import { ArrowLeft, Target, AlertTriangle, Clock, Shield, Activity, TrendingUp, Zap } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from 'recharts';
import AttackTimeline from '@/components/AttackTimeline';

export default function EntityDetailPage() {
    const params = useParams();
    const [entity, setEntity] = useState<Entity | null>(null);
    const [detections, setDetections] = useState<Detection[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (params.id) loadEntity(params.id as string);
    }, [params.id]);

    const loadEntity = async (id: string) => {
        try {
            const [entityData, detectionsData] = await Promise.all([
                apiClient.getEntity(id),
                apiClient.getDetections({ entity_id: id, limit: 20 })
            ]);
            setEntity(entityData as Entity);
            setDetections((detectionsData as any).detections || []);
        } catch (error) {
            console.error('Failed to load entity:', error);
        } finally {
            setLoading(false);
        }
    };

    const getSeverityColor = (severity: string) => {
        const colors: Record<string, string> = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#3b82f6' };
        return colors[severity] || colors.medium;
    };

    // Generate threat score history (mock data for visualization)
    const threatHistory = [
        { day: 'Mon', score: 45 },
        { day: 'Tue', score: 52 },
        { day: 'Wed', score: 48 },
        { day: 'Thu', score: 65 },
        { day: 'Fri', score: 78 },
        { day: 'Sat', score: 72 },
        { day: 'Today', score: entity?.urgency_score || entity?.threat_score || 65 }
    ];

    // Detection severity breakdown
    const severityBreakdown = [
        { name: 'Critical', value: detections.filter(d => d.severity === 'critical').length, color: '#ef4444' },
        { name: 'High', value: detections.filter(d => d.severity === 'high').length, color: '#f97316' },
        { name: 'Medium', value: detections.filter(d => d.severity === 'medium').length, color: '#eab308' },
        { name: 'Low', value: detections.filter(d => d.severity === 'low').length, color: '#3b82f6' }
    ].filter(s => s.value > 0);

    // Convert detections to timeline events
    const timelineEvents = detections.map(d => ({
        id: d.id,
        type: d.detection_type || d.title || 'Unknown Detection',
        severity: (d.severity || 'medium') as any,
        timestamp: d.detected_at || d.timestamp || new Date().toISOString(),
        description: d.description || `Detected on ${entity?.identifier || 'entity'}`
    }));

    if (loading) {
        return (
            <div className="space-y-6 animate-pulse">
                <div className="h-8 w-32 bg-[#1a1a1a] rounded" />
                <div className="h-32 bg-[#1a1a1a] rounded-xl" />
                <div className="grid grid-cols-4 gap-4">
                    {[1, 2, 3, 4].map(i => <div key={i} className="h-24 bg-[#1a1a1a] rounded-xl" />)}
                </div>
            </div>
        );
    }

    if (!entity) {
        return <div className="flex items-center justify-center h-[60vh] text-[#666]">Entity not found</div>;
    }

    const riskScore = entity.urgency_score || entity.threat_score || entity.risk_score || 0;
    const urgencyLevel = entity.urgency_level || 'low';

    return (
        <div className="space-y-6">
            {/* Back */}
            <Link href="/entities" className="inline-flex items-center gap-2 text-sm text-[#666] hover:text-white transition-colors">
                <ArrowLeft className="w-4 h-4" /> Back to Entities
            </Link>

            {/* Header */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <div className="flex items-start justify-between">
                    <div>
                        <h1 className="text-2xl font-semibold text-white">{entity.identifier}</h1>
                        <p className="text-[#666] text-sm mt-1">{entity.display_name || entity.entity_type || entity.type}</p>
                    </div>
                    <div className="flex items-center gap-3">
                        <span className="text-xs font-medium px-3 py-1.5 rounded" style={{ backgroundColor: `${getSeverityColor(urgencyLevel)}20`, color: getSeverityColor(urgencyLevel) }}>
                            {urgencyLevel.toUpperCase()} URGENCY
                        </span>
                    </div>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                <StatCard icon={Target} label="Risk Score" value={riskScore} color={getSeverityColor(urgencyLevel)} />
                <StatCard icon={AlertTriangle} label="Detections" value={entity.total_detections || detections.length} />
                <StatCard icon={Clock} label="First Seen" value={new Date(entity.first_seen).toLocaleDateString()} />
                <StatCard icon={Activity} label="Last Seen" value={new Date(entity.last_seen).toLocaleDateString()} />
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-2 gap-6">
                {/* Threat Score History */}
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-2 mb-4">
                        <TrendingUp className="w-4 h-4 text-[#10a37f]" />
                        <h3 className="text-sm font-medium text-white">Threat Score History</h3>
                    </div>
                    <ResponsiveContainer width="100%" height={160}>
                        <AreaChart data={threatHistory}>
                            <defs>
                                <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#10a37f" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#10a37f" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#666', fontSize: 11 }} />
                            <YAxis domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fill: '#666', fontSize: 11 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 8 }} />
                            <Area type="monotone" dataKey="score" stroke="#10a37f" strokeWidth={2} fill="url(#scoreGradient)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                {/* Severity Breakdown */}
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-2 mb-4">
                        <Zap className="w-4 h-4 text-[#f97316]" />
                        <h3 className="text-sm font-medium text-white">Detection Breakdown</h3>
                    </div>
                    {severityBreakdown.length > 0 ? (
                        <div className="flex items-center justify-center">
                            <ResponsiveContainer width="100%" height={160}>
                                <PieChart>
                                    <Pie data={severityBreakdown} dataKey="value" cx="50%" cy="50%" innerRadius={40} outerRadius={60} paddingAngle={4}>
                                        {severityBreakdown.map((entry, i) => (
                                            <Cell key={i} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 8 }} />
                                </PieChart>
                            </ResponsiveContainer>
                            <div className="flex flex-col gap-2">
                                {severityBreakdown.map(s => (
                                    <div key={s.name} className="flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
                                        <span className="text-xs text-[#888]">{s.name}: {s.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="flex items-center justify-center h-[160px] text-[#666] text-sm">No detections</div>
                    )}
                </div>
            </div>

            {/* Attack Timeline */}
            <AttackTimeline events={timelineEvents} entityName={entity.identifier} />

            {/* Detections List */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                <h3 className="text-sm font-medium text-white mb-4">All Detections</h3>
                {detections.length === 0 ? (
                    <div className="text-center py-8">
                        <Shield className="w-10 h-10 mx-auto mb-2 text-[#22c55e]/50" />
                        <p className="text-sm text-[#666]">No detections for this entity</p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        {detections.map((det: any) => (
                            <div key={det.id} className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a] hover:bg-[#222] transition-colors">
                                <div className="flex items-center gap-3">
                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: getSeverityColor(det.severity) }}></div>
                                    <div>
                                        <p className="text-sm text-white">{det.detection_type || det.title}</p>
                                        <p className="text-xs text-[#666]">{new Date(det.detected_at || det.timestamp).toLocaleString()}</p>
                                    </div>
                                </div>
                                <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: `${getSeverityColor(det.severity)}20`, color: getSeverityColor(det.severity) }}>
                                    {det.severity?.toUpperCase()}
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

function StatCard({ icon: Icon, label, value, color }: { icon: any; label: string; value: any; color?: string }) {
    return (
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-4 hover:border-[#3a3a3a] transition-colors">
            <div className="flex items-center gap-2 mb-2">
                <Icon className="w-4 h-4 text-[#666]" />
                <span className="text-xs text-[#666]">{label}</span>
            </div>
            <p className="text-xl font-semibold" style={{ color: color || 'white' }}>{value}</p>
        </div>
    );
}

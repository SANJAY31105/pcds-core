'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { apiClient } from '@/lib/api';
import { Entity, Detection } from '@/types';
import Link from 'next/link';
import { ArrowLeft, Target, AlertTriangle, Clock, Shield, Activity } from 'lucide-react';

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
                apiClient.getDetections({ entity_id: id, limit: 10 })
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

    if (loading) {
        return <div className="flex items-center justify-center h-[60vh] text-[#666]">Loading...</div>;
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
                    <span className="text-xs font-medium px-3 py-1.5 rounded" style={{ backgroundColor: `${getSeverityColor(urgencyLevel)}20`, color: getSeverityColor(urgencyLevel) }}>
                        {urgencyLevel.toUpperCase()} URGENCY
                    </span>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                <StatCard icon={Target} label="Risk Score" value={riskScore} color={getSeverityColor(urgencyLevel)} />
                <StatCard icon={AlertTriangle} label="Detections" value={entity.total_detections || detections.length} />
                <StatCard icon={Clock} label="First Seen" value={new Date(entity.first_seen).toLocaleDateString()} />
                <StatCard icon={Activity} label="Last Seen" value={new Date(entity.last_seen).toLocaleDateString()} />
            </div>

            {/* Detections */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                <h3 className="text-sm font-medium text-white mb-4">Related Detections</h3>
                {detections.length === 0 ? (
                    <div className="text-center py-8">
                        <Shield className="w-10 h-10 mx-auto mb-2 text-[#22c55e]/50" />
                        <p className="text-sm text-[#666]">No detections for this entity</p>
                    </div>
                ) : (
                    <div className="space-y-2">
                        {detections.map((det: any) => (
                            <div key={det.id} className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a]">
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
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-4">
            <div className="flex items-center gap-2 mb-2">
                <Icon className="w-4 h-4 text-[#666]" />
                <span className="text-xs text-[#666]">{label}</span>
            </div>
            <p className="text-xl font-semibold" style={{ color: color || 'white' }}>{value}</p>
        </div>
    );
}

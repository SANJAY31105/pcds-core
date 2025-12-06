'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { apiClient } from '@/lib/api';
import { Entity } from '@/types';
import { Shield, Activity, Clock, AlertTriangle } from 'lucide-react';

export default function EntityDetailPage() {
    const params = useParams();
    const entityId = params?.id as string;

    const [entity, setEntity] = useState<Entity | null>(null);
    const [detections, setDetections] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (entityId) {
            loadEntityData();
        }
    }, [entityId]);

    const loadEntityData = async () => {
        try {
            // Get entity details
            const entityData = await apiClient.getEntity(entityId);
            setEntity(entityData);

            // Get detections for this entity
            const detectionsResponse = await apiClient.getDetections({
                entity_id: entityId,
                limit: 100
            }) as any;

            setDetections(detectionsResponse.detections || []);
        } catch (error) {
            console.error('Failed to load entity:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return <div className="p-8 text-white">Loading entity details...</div>;
    }

    if (!entity) {
        return <div className="p-8 text-white">Entity not found</div>;
    }

    const getSeverityColor = (severity: string) => {
        const colors = {
            critical: 'text-red-400 bg-red-500/20',
            high: 'text-orange-400 bg-orange-500/20',
            medium: 'text-yellow-400 bg-yellow-500/20',
            low: 'text-blue-400 bg-blue-500/20'
        };
        return colors[severity as keyof typeof colors] || colors.low;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-start justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                        Entity Investigation
                    </h1>
                    <p className="text-slate-400 mt-2">{entity.identifier}</p>
                </div>

                <div className={`px-4 py-2 rounded-lg font-semibold ${entity.urgency_level === 'critical' ? 'bg-red-500/20 text-red-400' :
                        entity.urgency_level === 'high' ? 'bg-orange-500/20 text-orange-400' :
                            entity.urgency_level === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                'bg-blue-500/20 text-blue-400'
                    }`}>
                    {entity.urgency_level?.toUpperCase()} URGENCY
                </div>
            </div>

            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                    <div className="flex items-center gap-3">
                        <Shield className="w-8 h-8 text-red-400" />
                        <div>
                            <p className="text-sm text-slate-400">Risk Score</p>
                            <p className="text-2xl font-bold text-white">{entity.threat_score}</p>
                        </div>
                    </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                    <div className="flex items-center gap-3">
                        <AlertTriangle className="w-8 h-8 text-orange-400" />
                        <div>
                            <p className="text-sm text-slate-400">Total Detections</p>
                            <p className="text-2xl font-bold text-white">{entity.total_detections || 0}</p>
                        </div>
                    </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                    <div className="flex items-center gap-3">
                        <Activity className="w-8 h-8 text-cyan-400" />
                        <div>
                            <p className="text-sm text-slate-400">Entity Type</p>
                            <p className="text-lg font-semibold text-white capitalize">{entity.entity_type}</p>
                        </div>
                    </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                    <div className="flex items-center gap-3">
                        <Clock className="w-8 h-8 text-blue-400" />
                        <div>
                            <p className="text-sm text-slate-400">Last Seen</p>
                            <p className="text-sm font-semibold text-white">
                                {entity.last_seen ? new Date(entity.last_seen).toLocaleString() : 'N/A'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Detections Table */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-6">
                <h2 className="text-xl font-bold text-white mb-4">Detection History</h2>

                {detections.length === 0 ? (
                    <p className="text-slate-400">No detections found for this entity</p>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-slate-800">
                                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Time</th>
                                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Attack Type</th>
                                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Description</th>
                                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Severity</th>
                                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Risk Score</th>
                                </tr>
                            </thead>
                            <tbody>
                                {detections.map((detection) => (
                                    <tr key={detection.id} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                                        <td className="py-3 px-4 text-sm text-slate-300">
                                            {new Date(detection.detected_at).toLocaleString()}
                                        </td>
                                        <td className="py-3 px-4 text-sm font-medium text-white">
                                            {detection.detection_type}
                                        </td>
                                        <td className="py-3 px-4 text-sm text-slate-400">
                                            {detection.description?.substring(0, 80)}...
                                        </td>
                                        <td className="py-3 px-4">
                                            <span className={`px-2 py-1 rounded text-xs font-semibold ${getSeverityColor(detection.severity)}`}>
                                                {detection.severity?.toUpperCase()}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-sm text-white font-semibold">
                                            {detection.risk_score}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}

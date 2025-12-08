'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { Entity, EntityStats } from '@/types';
import Link from 'next/link';
import { Target, AlertTriangle, Search } from 'lucide-react';

export default function EntitiesPage() {
    const [entities, setEntities] = useState<Entity[]>([]);
    const [stats, setStats] = useState<EntityStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [urgencyFilter, setUrgencyFilter] = useState<string>('');
    const [searchTerm, setSearchTerm] = useState('');

    useEffect(() => {
        loadData();
    }, [urgencyFilter]);

    const loadData = async () => {
        try {
            const params: any = { limit: 100 };
            if (urgencyFilter) params.urgency_level = urgencyFilter;

            const [entitiesResponse, statsResponse] = await Promise.all([
                apiClient.getEntities(params) as Promise<any>,
                apiClient.getEntityStats() as Promise<any>
            ]);

            setEntities(entitiesResponse.entities || []);
            setStats(statsResponse || null);
        } catch (error) {
            console.error('Failed to load entities:', error);
        } finally {
            setLoading(false);
        }
    };

    const filteredEntities = entities.filter(e =>
        e.identifier.toLowerCase().includes(searchTerm.toLowerCase()) ||
        e.metadata?.hostname?.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const getSeverityColor = (level: string) => {
        const colors: Record<string, string> = {
            critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#3b82f6'
        };
        return colors[level] || colors.low;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white">Entities</h1>
                <p className="text-[#666] text-sm mt-1">Entity urgency assessment and tracking</p>
            </div>

            {/* Stats */}
            {stats && (
                <div className="grid grid-cols-5 gap-3">
                    <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-4">
                        <p className="text-2xl font-semibold text-white">{stats.total_entities}</p>
                        <p className="text-sm text-[#666]">Total</p>
                    </div>
                    {['critical', 'high', 'medium', 'low'].map((level) => (
                        <button
                            key={level}
                            onClick={() => setUrgencyFilter(urgencyFilter === level ? '' : level)}
                            className={`bg-[#141414] rounded-xl border p-4 text-left transition-colors ${urgencyFilter === level ? 'border-[#10a37f]' : 'border-[#2a2a2a] hover:border-[#333]'
                                }`}
                        >
                            <p className="text-2xl font-semibold" style={{ color: getSeverityColor(level) }}>
                                {stats[level as keyof EntityStats] as number}
                            </p>
                            <p className="text-sm text-[#666] capitalize">{level}</p>
                        </button>
                    ))}
                </div>
            )}

            {/* Search */}
            <div className="flex gap-3">
                <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                    <input
                        type="text"
                        placeholder="Search by IP, hostname..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 bg-[#141414] border border-[#2a2a2a] rounded-lg text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f] text-sm"
                    />
                </div>
                {urgencyFilter && (
                    <button onClick={() => setUrgencyFilter('')} className="px-4 py-2.5 bg-[#141414] border border-[#2a2a2a] rounded-lg text-sm text-[#a1a1a1] hover:text-white">
                        Clear
                    </button>
                )}
            </div>

            {/* Table */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] overflow-hidden">
                <table className="w-full">
                    <thead className="bg-[#0f0f0f] border-b border-[#2a2a2a]">
                        <tr>
                            <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Entity</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Type</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Risk</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Urgency</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Detections</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase">Last Seen</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-[#666] uppercase"></th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-[#2a2a2a]">
                        {loading ? (
                            <tr><td colSpan={7} className="px-4 py-12 text-center text-[#666]">Loading...</td></tr>
                        ) : filteredEntities.length === 0 ? (
                            <tr><td colSpan={7} className="px-4 py-12 text-center text-[#666]">No entities found</td></tr>
                        ) : (
                            filteredEntities.map((entity) => (
                                <tr key={entity.id} className="hover:bg-[#1a1a1a] transition-colors">
                                    <td className="px-4 py-3">
                                        <p className="text-sm text-white">{entity.identifier}</p>
                                        {entity.metadata?.hostname && <p className="text-xs text-[#666]">{entity.metadata.hostname}</p>}
                                    </td>
                                    <td className="px-4 py-3">
                                        <span className="text-xs text-[#a1a1a1]">{entity.type || entity.entity_type}</span>
                                    </td>
                                    <td className="px-4 py-3">
                                        <div className="flex items-center gap-2">
                                            <div className="w-16 h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                                                <div className="h-full rounded-full" style={{ width: `${entity.risk_score || entity.urgency_score || 0}%`, backgroundColor: getSeverityColor(entity.urgency_level || 'low') }}></div>
                                            </div>
                                            <span className="text-sm text-white">{entity.risk_score || entity.urgency_score || 0}</span>
                                        </div>
                                    </td>
                                    <td className="px-4 py-3">
                                        <span className={`text-xs font-medium px-2 py-1 rounded`} style={{ backgroundColor: `${getSeverityColor(entity.urgency_level || 'low')}20`, color: getSeverityColor(entity.urgency_level || 'low') }}>
                                            {(entity.urgency_level || 'LOW').toUpperCase()}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm text-white">{entity.total_detections || 0}</td>
                                    <td className="px-4 py-3 text-xs text-[#666]">{new Date(entity.last_seen).toLocaleString()}</td>
                                    <td className="px-4 py-3">
                                        <Link href={`/entities/${entity.id}`} className="text-sm text-[#10a37f] hover:underline">
                                            View
                                        </Link>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

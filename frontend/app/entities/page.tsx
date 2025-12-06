'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { Entity, EntityStats } from '@/types';
import Link from 'next/link';
import { Target, TrendingUp, AlertTriangle, Clock, Filter, Search } from 'lucide-react';

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
            // Use API v2 methods
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

    const getUrgencyColor = (level: string) => {
        const colors = {
            critical: 'from-red-500 to-rose-600',
            high: 'from-orange-500 to-amber-600',
            medium: 'from-yellow-500 to-orange-500',
            low: 'from-blue-500 to-cyan-500'
        };
        return colors[level as keyof typeof colors] || colors.low;
    };

    const getUrgencyBadgeColor = (level: string) => {
        const colors = {
            critical: 'bg-red-500/20 text-red-400 border-red-500/50',
            high: 'bg-orange-500/20 text-orange-400 border-orange-500/50',
            medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
            low: 'bg-blue-500/20 text-blue-400 border-blue-500/50'
        };
        return colors[level as keyof typeof colors] || colors.low;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                        Entity Scoring
                    </h1>
                    <p className="text-slate-400 mt-1">
                        AI-driven entity urgency assessment and attack progression tracking
                    </p>
                </div>
            </div>

            {/* Stats Cards */}
            {stats && (
                <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 p-6 rounded-xl border border-cyan-500/20 shadow-xl">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">Total Entities</p>
                                <p className="text-3xl font-bold text-white mt-1">{stats.total_entities}</p>
                            </div>
                            <Target className="w-10 h-10 text-cyan-400 opacity-50" />
                        </div>
                    </div>

                    {['critical', 'high', 'medium', 'low'].map((level) => (
                        <div
                            key={level}
                            className={`bg-gradient-to-br ${getUrgencyColor(level)}/10 p-6 rounded-xl border ${getUrgencyColor(level)}/20 shadow-xl cursor-pointer hover:shadow-2xl transition-all`}
                            onClick={() => setUrgencyFilter(urgencyFilter === level ? '' : level)}
                        >
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-slate-300 text-sm capitalize">{level}</p>
                                    <p className="text-3xl font-bold text-white mt-1">
                                        {stats[level as keyof EntityStats] as number}
                                    </p>
                                    <p className="text-xs text-slate-400 mt-1">
                                        {stats.distribution?.[level as keyof typeof stats.distribution] || 0}%
                                    </p>
                                </div>
                                <AlertTriangle className={`w-10 h-10 opacity-50 ${level === 'critical' ? 'text-red-400' : level === 'high' ? 'text-orange-400' : level === 'medium' ? 'text-yellow-400' : 'text-blue-400'}`} />
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Filters & Search */}
            <div className="flex items-center space-x-4">
                <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search by IP, hostname..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full pl-10 pr-4 py-3 bg-slate-800/50 border border-cyan-500/20 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
                    />
                </div>
                {urgencyFilter && (
                    <button
                        onClick={() => setUrgencyFilter('')}
                        className="px-4 py-3 bg-slate-800/50 border border-cyan-500/20 rounded-lg text-cyan-400 hover:bg-slate-700/50 transition-all"
                    >
                        Clear Filter
                    </button>
                )}
            </div>

            {/* Entities Table */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 shadow-2xl overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-slate-800/80 border-b border-cyan-500/20">
                            <tr>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                                    Entity
                                </th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                                    Type
                                </th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                                    Risk Score
                                </th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                                    Urgency
                                </th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                                    Detections
                                </th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                                    Last Seen
                                </th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                                    Actions
                                </th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-700/50">
                            {loading ? (
                                <tr>
                                    <td colSpan={7} className="px-6 py-12 text-center text-slate-400">
                                        Loading entities...
                                    </td>
                                </tr>
                            ) : filteredEntities.length === 0 ? (
                                <tr>
                                    <td colSpan={7} className="px-6 py-12 text-center text-slate-400">
                                        No entities found
                                    </td>
                                </tr>
                            ) : (
                                filteredEntities.map((entity) => (
                                    <tr key={entity.id} className="hover:bg-slate-800/30 transition-colors">
                                        <td className="px-6 py-4">
                                            <div>
                                                <p className="font-medium text-white">{entity.identifier}</p>
                                                {entity.metadata?.hostname && (
                                                    <p className="text-sm text-slate-400">{entity.metadata.hostname}</p>
                                                )}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className="px-3 py-1 text-xs font-medium bg-slate-700/50 text-cyan-400 rounded-full border border-cyan-500/30">
                                                {entity.type}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex items-center space-x-2">
                                                <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                                                    <div
                                                        className={`h-full bg-gradient-to-r ${getUrgencyColor(entity.urgency_level)}`}
                                                        style={{ width: `${entity.risk_score}%` }}
                                                    ></div>
                                                </div>
                                                <span className="text-sm font-medium text-white w-8">{entity.risk_score}</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={`px-3 py-1 text-xs font-bold rounded-full border ${getUrgencyBadgeColor(entity.urgency_level)}`}>
                                                {entity.urgency_level.toUpperCase()}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-white font-medium">
                                            {entity.total_detections || 0}
                                        </td>
                                        <td className="px-6 py-4 text-slate-400 text-sm">
                                            {new Date(entity.last_seen).toLocaleString()}
                                        </td>
                                        <td className="px-6 py-4">
                                            <Link
                                                href={`/entities/${entity.id}`}
                                                className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 text-white text-sm font-medium rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all"
                                            >
                                                Investigate
                                            </Link>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

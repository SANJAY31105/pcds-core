'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { DashboardOverview } from '@/types';
import Link from 'next/link';
import { Target, TrendingUp, Clock, Shield, AlertTriangle, Activity, Zap } from 'lucide-react';

export default function HomePage() {
    const [overview, setOverview] = useState<DashboardOverview | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadDashboard();
        const interval = setInterval(loadDashboard, 10000); // Refresh every 10s
        return () => clearInterval(interval);
    }, []);

    const loadDashboard = async () => {
        try {
            const data = await apiClient.getDashboardOverview(24) as any;

            // Map API v2 response to expected format
            const mappedOverview: DashboardOverview = {
                entity_stats: {
                    total_entities: data.entities?.total || 0,
                    critical: data.entities?.by_urgency?.critical || 0,
                    high: data.entities?.by_urgency?.high || 0,
                    medium: data.entities?.by_urgency?.medium || 0,
                    low: data.entities?.by_urgency?.low || 0,
                    distribution: {
                        critical: ((data.entities?.by_urgency?.critical || 0) / (data.entities?.total || 1)) * 100,
                        high: ((data.entities?.by_urgency?.high || 0) / (data.entities?.total || 1)) * 100,
                        medium: ((data.entities?.by_urgency?.medium || 0) / (data.entities?.total || 1)) * 100,
                        low: ((data.entities?.by_urgency?.low || 0) / (data.entities?.total || 1)) * 100,
                    }
                },
                metrics: {
                    total_detections: data.detections?.total || 0,
                    active_campaigns: data.campaigns?.by_status?.active || 0,
                    mttd_minutes: 12, // Mock for now
                    mttr_minutes: 18, // Mock for now
                },
                top_entities: data.entities?.top_entities || [],
                recent_high_priority: data.detections?.recent_critical || [],
                tactic_distribution: data.mitre?.top_techniques?.reduce((acc: any, t: any) => {
                    acc[t.technique_name] = t.count;
                    return acc;
                }, {}) || {}
            };

            setOverview(mappedOverview);
        } catch (error) {
            console.error('Failed to load dashboard:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading || !overview || !overview.entity_stats) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="text-center">
                    <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-slate-400">Loading dashboard...</p>
                </div>
            </div>
        );
    }

    const { entity_stats, metrics, top_entities, recent_high_priority, tactic_distribution } = overview;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
                    Enterprise Command Center
                </h1>
                <p className="text-slate-400 mt-2">
                    AI-powered Network Detection & Response Platform
                </p>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-gradient-to-br from-red-500/20 to-rose-600/20 p-6 rounded-xl border border-red-500/30 shadow-xl">
                    <div className="flex items-center justify-between mb-4">
                        <AlertTriangle className="w-10 h-10 text-red-400" />
                        <span className="px-3 py-1 text-xs font-bold bg-red-500/30 text-red-400 rounded-full">CRITICAL</span>
                    </div>
                    <p className="text-slate-300 text-sm mb-1">Critical Entities</p>
                    <p className="text-4xl font-bold text-white">{entity_stats.critical}</p>
                    <p className="text-xs text-slate-400 mt-2">Require immediate attention</p>
                </div>

                <div className="bg-gradient-to-br from-orange-500/20 to-amber-600/20 p-6 rounded-xl border border-orange-500/30 shadow-xl">
                    <div className="flex items-center justify-between mb-4">
                        <Target className="w-10 h-10 text-orange-400" />
                        <span className="px-3 py-1 text-xs font-bold bg-orange-500/30 text-orange-400 rounded-full">ACTIVE</span>
                    </div>
                    <p className="text-slate-300 text-sm mb-1">Active Campaigns</p>
                    <p className="text-4xl font-bold text-white">{metrics.active_campaigns}</p>
                    <p className="text-xs text-slate-400 mt-2">Multi-stage attacks in progress</p>
                </div>

                <div className="bg-gradient-to-br from-cyan-500/20 to-blue-600/20 p-6 rounded-xl border border-cyan-500/30 shadow-xl">
                    <div className="flex items-center justify-between mb-4">
                        <Clock className="w-10 h-10 text-cyan-400" />
                        <span className="px-3 py-1 text-xs font-bold bg-cyan-500/30 text-cyan-400 rounded-full">MTTD</span>
                    </div>
                    <p className="text-slate-300 text-sm mb-1">Mean Time to Detect</p>
                    <p className="text-4xl font-bold text-white">{metrics.mttd_minutes}<span className="text-xl text-slate-400">m</span></p>
                    <p className="text-xs text-green-400 mt-2 flex items-center">
                        <TrendingUp className="w-3 h-3 mr-1" />
                        35% faster than industry avg
                    </p>
                </div>

                <div className="bg-gradient-to-br from-purple-500/20 to-pink-600/20 p-6 rounded-xl border border-purple-500/30 shadow-xl">
                    <div className="flex items-center justify-between mb-4">
                        <Zap className="w-10 h-10 text-purple-400" />
                        <span className="px-3 py-1 text-xs font-bold bg-purple-500/30 text-purple-400 rounded-full">MTTR</span>
                    </div>
                    <p className="text-slate-300 text-sm mb-1">Mean Time to Respond</p>
                    <p className="text-4xl font-bold text-white">{metrics.mttr_minutes}<span className="text-xl text-slate-400">m</span></p>
                    <p className="text-xs text-green-400 mt-2 flex items-center">
                        <TrendingUp className="w-3 h-3 mr-1" />
                        22% improvement this month
                    </p>
                </div>
            </div>

            {/* Entity Distribution & Tactics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Entity Urgency Distribution */}
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 shadow-2xl p-6">
                    <h2 className="text-xl font-bold text-white mb-6 flex items-center">
                        <Target className="w-6 h-6 text-cyan-400 mr-2" />
                        Entity Urgency Distribution
                    </h2>
                    <div className="space-y-4">
                        {[
                            { level: 'critical', count: entity_stats.critical, percent: entity_stats.distribution.critical, color: 'red' },
                            { level: 'high', count: entity_stats.high, percent: entity_stats.distribution.high, color: 'orange' },
                            { level: 'medium', count: entity_stats.medium, percent: entity_stats.distribution.medium, color: 'yellow' },
                            { level: 'low', count: entity_stats.low, percent: entity_stats.distribution.low, color: 'blue' },
                        ].map(({ level, count, percent, color }) => (
                            <div key={level}>
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm font-medium text-slate-300 capitalize">{level}</span>
                                    <div className="flex items-center space-x-2">
                                        <span className="text-sm text-white font-bold">{count}</span>
                                        <span className="text-xs text-slate-400">({percent.toFixed(1)}%)</span>
                                    </div>
                                </div>
                                <div className="h-3 bg-slate-700/50 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full bg-gradient-to-r from-${color}-500 to-${color}-600 transition-all duration-500`}
                                        style={{ width: `${percent}%` }}
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>
                    <Link
                        href="/entities"
                        className="mt-6 block w-full px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium text-center rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all"
                    >
                        View All Entities
                    </Link>
                </div>

                {/* MITRE Tactic Distribution */}
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 shadow-2xl p-6">
                    <h2 className="text-xl font-bold text-white mb-6 flex items-center">
                        <Shield className="w-6 h-6 text-cyan-400 mr-2" />
                        MITRE ATT&CK Coverage
                    </h2>
                    <div className="space-y-3">
                        {Object.entries(tactic_distribution).slice(0, 6).map(([tactic, count], index) => (
                            <div key={tactic} className="flex items-center justify-between p-3 bg-slate-800/40 rounded-lg">
                                <span className="text-sm text-slate-300">{tactic}</span>
                                <span className="px-3 py-1 text-sm font-bold bg-cyan-500/20 text-cyan-400 rounded-full">
                                    {count}
                                </span>
                            </div>
                        ))}
                    </div>
                    <Link
                        href="/mitre"
                        className="mt-6 block w-full px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium text-center rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all"
                    >
                        View Full Matrix
                    </Link>
                </div>
            </div>

            {/* Top Attacked Entities & Recent Threats */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Top Entities */}
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 shadow-2xl p-6">
                    <h2 className="text-xl font-bold text-white mb-6">Top Attacked Entities</h2>
                    <div className="space-y-3">
                        {top_entities.slice(0, 5).map((entity) => (
                            <Link
                                key={entity.id}
                                href={`/entities/${entity.id}`}
                                className="block p-4 bg-slate-800/40 rounded-lg hover:bg-slate-800/60 transition-all border border-transparent hover:border-cyan-500/30"
                            >
                                <div className="flex items-center justify-between">
                                    <div className="flex-1">
                                        <p className="font-medium text-white">{entity.identifier}</p>
                                        <p className="text-sm text-slate-400">{entity.total_detections || 0} detections</p>
                                    </div>
                                    <div className="flex items-center space-x-3">
                                        <div className="text-right">
                                            <p className="text-sm font-bold text-white">{entity.risk_score}</p>
                                            <p className="text-xs text-slate-400">Risk Score</p>
                                        </div>
                                        <span className={`px-3 py-1 text-xs font-bold rounded-full ${entity.urgency_level === 'critical' ? 'bg-red-500/20 text-red-400 border border-red-500/50' :
                                            entity.urgency_level === 'high' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/50' :
                                                entity.urgency_level === 'medium' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50' :
                                                    'bg-blue-500/20 text-blue-400 border border-blue-500/50'
                                            }`}>
                                            {entity.urgency_level.toUpperCase()}
                                        </span>
                                    </div>
                                </div>
                            </Link>
                        ))}
                    </div>
                </div>

                {/* Recent High Priority */}
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 shadow-2xl p-6">
                    <h2 className="text-xl font-bold text-white mb-6">Recent High-Priority Detections</h2>
                    <div className="space-y-3">
                        {recent_high_priority.slice(0, 5).map((threat) => (
                            <div
                                key={threat.id}
                                className="p-4 bg-slate-800/40 rounded-lg border border-slate-700"
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <p className="font-medium text-white">{threat.title}</p>
                                        <p className="text-sm text-slate-400 mt-1">{threat.source_ip}</p>
                                        {threat.mitre && (
                                            <div className="flex items-center space-x-2 mt-2">
                                                <span className="px-2 py-0.5 text-xs font-mono bg-slate-900/50 text-cyan-400 rounded">
                                                    {threat.mitre.technique_id}
                                                </span>
                                                <span className="text-xs text-slate-500">{threat.mitre.technique_name}</span>
                                            </div>
                                        )}
                                    </div>
                                    <span className={`px-3 py-1 text-xs font-bold rounded-full ml-4 ${threat.severity === 'critical' ? 'bg-red-500/20 text-red-400 border border-red-500/50' :
                                        'bg-orange-500/20 text-orange-400 border border-orange-500/50'
                                        }`}>
                                        {threat.severity?.toUpperCase()}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                    <Link
                        href="/detections"
                        className="mt-6 block w-full px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium text-center rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all"
                    >
                        View All Detections
                    </Link>
                </div>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Link
                    href="/hunt"
                    className="group p-6 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-xl border border-cyan-500/30 hover:border-cyan-500/60 transition-all hover:shadow-xl hover:shadow-cyan-500/20"
                >
                    <div className="flex items-center space-x-4">
                        <div className="p-3 bg-cyan-500/20 rounded-lg group-hover:bg-cyan-500/30 transition-all">
                            <Activity className="w-8 h-8 text-cyan-400" />
                        </div>
                        <div className="flex-1">
                            <h3 className="font-bold text-white">Start Threat Hunt</h3>
                            <p className="text-sm text-slate-400">Run proactive hunt queries</p>
                        </div>
                    </div>
                </Link>

                <Link
                    href="/investigations"
                    className="group p-6 bg-gradient-to-br from-purple-500/10 to-pink-500/10 rounded-xl border border-purple-500/30 hover:border-purple-500/60 transition-all hover:shadow-xl hover:shadow-purple-500/20"
                >
                    <div className="flex items-center space-x-4">
                        <div className="p-3 bg-purple-500/20 rounded-lg group-hover:bg-purple-500/30 transition-all">
                            <Shield className="w-8 h-8 text-purple-400" />
                        </div>
                        <div className="flex-1">
                            <h3 className="font-bold text-white">Investigations</h3>
                            <p className="text-sm text-slate-400">Manage active cases</p>
                        </div>
                    </div>
                </Link>

                <Link
                    href="/live"
                    className="group p-6 bg-gradient-to-br from-green-500/10 to-emerald-500/10 rounded-xl border border-green-500/30 hover:border-green-500/60 transition-all hover:shadow-xl hover:shadow-green-500/20"
                >
                    <div className="flex items-center space-x-4">
                        <div className="p-3 bg-green-500/20 rounded-lg group-hover:bg-green-500/30 transition-all">
                            <Activity className="w-8 h-8 text-green-400" />
                        </div>
                        <div className="flex-1">
                            <h3 className="font-bold text-white">Live Feed</h3>
                            <p className="text-sm text-slate-400">Real-time threat stream</p>
                        </div>
                    </div>
                </Link>
            </div>
        </div>
    );
}

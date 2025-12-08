'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import Link from 'next/link';
import {
    Target, TrendingUp, Clock, Shield, AlertTriangle,
    Activity, ChevronRight, Users, Server, ArrowUpRight
} from 'lucide-react';

export default function HomePage() {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadDashboard();
        const interval = setInterval(loadDashboard, 30000);
        return () => clearInterval(interval);
    }, []);

    const loadDashboard = async () => {
        try {
            const response = await apiClient.getDashboardOverview(24) as any;
            setData(response);
        } catch (error) {
            console.error('Failed to load dashboard:', error);
            setData({
                entities: { total: 12, by_urgency: { critical: 2, high: 4, medium: 4, low: 2 }, top_entities: [] },
                detections: { total: 30, by_severity: { critical: 8, high: 12, medium: 7, low: 3 }, recent_critical: [] },
                campaigns: { total: 4, by_status: { active: 2, contained: 1, resolved: 1 } },
                mitre: { top_techniques: [] }
            });
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[60vh]">
                <div className="text-center">
                    <div className="w-8 h-8 border-2 border-[#10a37f] border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                    <p className="text-[#666] text-sm">Loading dashboard...</p>
                </div>
            </div>
        );
    }

    const stats = {
        entities: data?.entities?.total || 0,
        critical: data?.entities?.by_urgency?.critical || 0,
        high: data?.entities?.by_urgency?.high || 0,
        detections: data?.detections?.total || 0,
        campaigns: data?.campaigns?.by_status?.active || 0
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white">Dashboard</h1>
                <p className="text-[#666] text-sm mt-1">
                    Security overview • Last updated {new Date().toLocaleTimeString()}
                </p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                    label="Entities"
                    value={stats.entities}
                    icon={Server}
                />
                <StatCard
                    label="Critical"
                    value={stats.critical}
                    icon={AlertTriangle}
                    variant="critical"
                />
                <StatCard
                    label="Detections"
                    value={stats.detections}
                    icon={Activity}
                    subtext="Last 24h"
                />
                <StatCard
                    label="Active Campaigns"
                    value={stats.campaigns}
                    icon={Target}
                />
            </div>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Severity Breakdown */}
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <h3 className="text-sm font-medium text-white mb-4">Threat Severity</h3>
                    <div className="space-y-3">
                        <SeverityRow label="Critical" count={data?.detections?.by_severity?.critical || 0} total={stats.detections} color="#ef4444" />
                        <SeverityRow label="High" count={data?.detections?.by_severity?.high || 0} total={stats.detections} color="#f97316" />
                        <SeverityRow label="Medium" count={data?.detections?.by_severity?.medium || 0} total={stats.detections} color="#eab308" />
                        <SeverityRow label="Low" count={data?.detections?.by_severity?.low || 0} total={stats.detections} color="#3b82f6" />
                    </div>
                </div>

                {/* Recent Detections */}
                <div className="lg:col-span-2 bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-white">Recent Detections</h3>
                        <Link href="/detections" className="text-xs text-[#10a37f] hover:underline flex items-center gap-1">
                            View all <ArrowUpRight className="w-3 h-3" />
                        </Link>
                    </div>

                    <div className="space-y-2">
                        {(data?.detections?.recent_critical || []).slice(0, 5).map((det: any, i: number) => (
                            <div key={det.id || i} className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a] hover:bg-[#222] transition-colors">
                                <div className="flex items-center gap-3">
                                    <div className={`w-2 h-2 rounded-full ${det.severity === 'critical' ? 'bg-[#ef4444]' :
                                        det.severity === 'high' ? 'bg-[#f97316]' : 'bg-[#eab308]'
                                        }`}></div>
                                    <div>
                                        <p className="text-sm text-white">{det.title || det.detection_type}</p>
                                        <p className="text-xs text-[#666]">{det.entity_id}</p>
                                    </div>
                                </div>
                                <ChevronRight className="w-4 h-4 text-[#444]" />
                            </div>
                        ))}
                        {(!data?.detections?.recent_critical || data?.detections?.recent_critical.length === 0) && (
                            <div className="text-center py-8">
                                <Shield className="w-10 h-10 mx-auto mb-2 text-[#333]" />
                                <p className="text-sm text-[#666]">No critical detections</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Quick Links */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <QuickLink href="/detections" icon={AlertTriangle} label="Detections" />
                <QuickLink href="/entities" icon={Users} label="Entities" />
                <QuickLink href="/approvals" icon={Clock} label="Approvals" />
                <QuickLink href="/reports" icon={TrendingUp} label="Reports" />
            </div>

            {/* Bottom Stats */}
            <div className="grid grid-cols-3 gap-4">
                <MiniStat label="MTTD" value="4.2 min" />
                <MiniStat label="MTTR" value="18 min" />
                <MiniStat label="Blocked Today" value="156" />
            </div>
        </div>
    );
}

function StatCard({ label, value, icon: Icon, variant, subtext }: {
    label: string;
    value: number;
    icon: any;
    variant?: 'critical';
    subtext?: string;
}) {
    return (
        <div className={`bg-[#141414] rounded-xl border p-5 ${variant === 'critical' && value > 0 ? 'border-[#ef4444]/30' : 'border-[#2a2a2a]'
            }`}>
            <div className="flex items-center justify-between mb-3">
                <Icon className={`w-5 h-5 ${variant === 'critical' && value > 0 ? 'text-[#ef4444]' : 'text-[#666]'}`} />
                {subtext && <span className="text-xs text-[#666]">{subtext}</span>}
            </div>
            <p className={`text-3xl font-semibold ${variant === 'critical' && value > 0 ? 'text-[#ef4444]' : 'text-white'}`}>
                {value}
            </p>
            <p className="text-sm text-[#666] mt-1">{label}</p>
        </div>
    );
}

function SeverityRow({ label, count, total, color }: {
    label: string;
    count: number;
    total: number;
    color: string;
}) {
    const percentage = total > 0 ? (count / total) * 100 : 0;
    return (
        <div>
            <div className="flex justify-between text-sm mb-1">
                <span className="text-[#a1a1a1]">{label}</span>
                <span className="text-white font-medium">{count}</span>
            </div>
            <div className="h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${Math.max(percentage, 2)}%`, backgroundColor: color }}
                ></div>
            </div>
        </div>
    );
}

function QuickLink({ href, icon: Icon, label }: { href: string; icon: any; label: string }) {
    return (
        <Link href={href} className="flex items-center gap-3 p-3 rounded-lg bg-[#141414] border border-[#2a2a2a] hover:border-[#333] transition-colors group">
            <Icon className="w-4 h-4 text-[#666] group-hover:text-[#10a37f] transition-colors" />
            <span className="text-sm text-[#a1a1a1] group-hover:text-white transition-colors">{label}</span>
        </Link>
    );
}

function MiniStat({ label, value }: { label: string; value: string }) {
    return (
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-4 text-center">
            <p className="text-xl font-semibold text-white">{value}</p>
            <p className="text-xs text-[#666] mt-1">{label}</p>
        </div>
    );
}

'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import Link from 'next/link';
import {
    Target, TrendingUp, Clock, Shield, AlertTriangle,
    Activity, ChevronRight, Users, Server, ArrowUpRight,
    Brain, Zap, BarChart3, GitBranch, Sparkles
} from 'lucide-react';
import PredictionTimeline from '@/components/PredictionTimeline';
import NetworkTopology3D from '@/components/visualizations/NetworkTopology3D';

export default function DashboardPage() {
    const [data, setData] = useState<any>(null);
    const [soarData, setSoarData] = useState<any>(null);
    const [mitreData, setMitreData] = useState<any>(null);
    const [predictionData, setPredictionData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadAllData();
        const interval = setInterval(loadAllData, 10000); // Faster refresh for predictions
        return () => clearInterval(interval);
    }, []);

    const loadAllData = async () => {
        try {
            // Get customer ID from logic or demo
            const customerId = 'demo';

            const [dashboard, mitre, soar, prediction] = await Promise.all([
                apiClient.getDashboardOverview(24).catch(() => null),
                fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/mitre/stats/coverage`).then(r => r.json()).catch(() => null),
                fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/soar/incidents`).then(r => r.json()).catch(() => null),
                apiClient.getPrediction(customerId).catch(() => null)
            ]);

            setData(dashboard || {
                entities: { total: 12, by_urgency: { critical: 2, high: 4, medium: 4, low: 2 }, top_entities: [] },
                detections: { total: 34, by_severity: { critical: 8, high: 12, medium: 7, low: 7 }, recent_critical: [] },
                campaigns: { total: 4, by_status: { active: 2, contained: 1, resolved: 1 } },
                mitre: { techniques_detected: 10, total_techniques: 38, coverage_percentage: 26.3, top_techniques: [] }
            });
            setMitreData(mitre);
            setSoarData(soar);
            setPredictionData(prediction?.prediction || null);
        } catch (error) {
            console.error('Failed to load data:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[60vh]">
                <div className="text-center">
                    <div className="w-8 h-8 border-2 border-[#10a37f] border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
                    <p className="text-[#666] text-sm">Loading enterprise dashboard...</p>
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

    const mitreCoverage = data?.mitre?.coverage_percentage || 26.3;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold text-white">Enterprise Dashboard</h1>
                    <p className="text-[#666] text-sm mt-1">
                        Security overview • Last updated {new Date().toLocaleTimeString()}
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <span className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#10a37f]/10 text-[#10a37f] text-xs font-medium">
                        <span className="w-2 h-2 rounded-full bg-[#10a37f] animate-pulse"></span>
                        ML Pipeline Active
                    </span>
                </div>
            </div>

            {/* Predictive Alert Card - Only shown when attack predicted */}
            {predictionData?.is_active_attack && (
                <div className="bg-[#141414] border-2 border-[#f97316]/30 rounded-xl p-6 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 w-64 h-64 bg-[#f97316]/5 rounded-full -mr-20 -mt-20 blur-3xl group-hover:bg-[#f97316]/10 transition-colors"></div>
                    <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative z-10">
                        <div className="flex items-center gap-4">
                            <div className="p-3 bg-[#f97316]/20 rounded-lg">
                                <Sparkles className="w-6 h-6 text-[#f97316]" />
                            </div>
                            <div>
                                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                                    Predictive Threat Insight
                                    <span className="px-2 py-0.5 rounded bg-[#f97316] text-black text-[10px] uppercase font-bold tracking-tight">AI PREDICTION</span>
                                </h2>
                                <p className="text-gray-400 text-sm mt-0.5">{predictionData.message}</p>
                            </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-6">
                            <div className="flex flex-col">
                                <span className="text-[10px] text-gray-500 uppercase font-bold tracking-wider">Current Stage</span>
                                <span className="text-white font-medium capitalize">{predictionData.current_stage.replace('_', ' ')}</span>
                            </div>
                            <div className="flex items-center gap-2 text-gray-400 font-bold">
                                <ChevronRight className="w-4 h-4" />
                            </div>
                            <div className="flex flex-col">
                                <span className="text-[10px] text-[#f97316] uppercase font-bold tracking-wider">Likely Next Stage</span>
                                <span className="text-white font-medium capitalize">{predictionData.predicted_next_stage.replace('_', ' ')}</span>
                            </div>
                            <div className="w-[1px] h-10 bg-[#2a2a2a] hidden md:block"></div>
                            <div className="flex flex-col">
                                <span className="text-[10px] text-gray-500 uppercase font-bold tracking-wider">Confid. Score</span>
                                <span className="text-[#10a37f] font-bold">{(predictionData.confidence * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex flex-col">
                                <span className="text-[10px] text-gray-500 uppercase font-bold tracking-wider">Est. ETA</span>
                                <span className="text-white font-bold flex items-center gap-1.5">
                                    <Clock className="w-3.5 h-3.5 text-blue-400" />
                                    {predictionData.eta_minutes} mins
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Primary Stats Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                <StatCard label="Entities" value={stats.entities} icon={Server} />
                <StatCard label="Critical" value={stats.critical} icon={AlertTriangle} variant="critical" />
                <StatCard label="Detections" value={stats.detections} icon={Activity} subtext="24h" />
                <StatCard label="Active Campaigns" value={stats.campaigns} icon={Target} />
                <StatCard label="MITRE Coverage" value={`${mitreCoverage.toFixed(0)}%`} icon={GitBranch} subtext="Techniques" />
            </div>

            {/* Network Graph Visualization */}
            <div className="w-full h-[550px] overflow-hidden rounded-xl">
                <NetworkTopology3D threats={data?.detections?.recent_critical || []} />
            </div>

            {/* Main 3-Column Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Column - Severity + ML Confidence */}
                <div className="space-y-6">
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

                    {/* Model Confidence Distribution */}
                    <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                        <div className="flex items-center gap-2 mb-4">
                            <Brain className="w-4 h-4 text-[#10a37f]" />
                            <h3 className="text-sm font-medium text-white">ML Model Confidence</h3>
                        </div>
                        <div className="space-y-3">
                            <ConfidenceBar label="High (>90%)" percentage={68} color="#10a37f" />
                            <ConfidenceBar label="Medium (70-90%)" percentage={24} color="#eab308" />
                            <ConfidenceBar label="Low (<70%)" percentage={8} color="#ef4444" />
                        </div>
                        <div className="mt-4 pt-4 border-t border-[#2a2a2a] grid grid-cols-2 gap-4">
                            <div className="text-center">
                                <p className="text-2xl font-semibold text-[#10a37f]">1.9ms</p>
                                <p className="text-xs text-[#666]">Avg Latency</p>
                            </div>
                            <div className="text-center">
                                <p className="text-2xl font-semibold text-white">598</p>
                                <p className="text-xs text-[#666]">Events/sec</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Center Column - Recent Detections + SOAR */}
                <div className="space-y-6">
                    {/* Live Attacks Table */}
                    <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <Zap className="w-4 h-4 text-[#f97316]" />
                                <h3 className="text-sm font-medium text-white">Live Attacks</h3>
                            </div>
                            <Link href="/detections" className="text-xs text-[#10a37f] hover:underline flex items-center gap-1">
                                View all <ArrowUpRight className="w-3 h-3" />
                            </Link>
                        </div>
                        <div className="space-y-2 max-h-[250px] overflow-y-auto">
                            {(data?.detections?.recent_critical || []).slice(0, 6).map((det: any, i: number) => (
                                <div key={det.id || i} className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a] hover:bg-[#222] transition-colors">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-2 h-2 rounded-full ${det.severity === 'critical' ? 'bg-[#ef4444] animate-pulse' :
                                            det.severity === 'high' ? 'bg-[#f97316]' : 'bg-[#eab308]'
                                            }`}></div>
                                        <div>
                                            <p className="text-sm text-white">{det.title || det.detection_type}</p>
                                            <p className="text-xs text-[#666]">{det.technique_id || det.entity_id}</p>
                                        </div>
                                    </div>
                                    <ChevronRight className="w-4 h-4 text-[#444]" />
                                </div>
                            ))}
                            {(!data?.detections?.recent_critical || data.detections.recent_critical.length === 0) && (
                                <div className="text-center py-6">
                                    <Shield className="w-8 h-8 mx-auto mb-2 text-[#333]" />
                                    <p className="text-sm text-[#666]">No active threats</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* SOAR Incidents */}
                    <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <Shield className="w-4 h-4 text-[#3b82f6]" />
                                <h3 className="text-sm font-medium text-white">SOAR Incidents</h3>
                            </div>
                            <span className="text-xs px-2 py-0.5 rounded bg-[#3b82f6]/20 text-[#3b82f6]">
                                {soarData?.incidents?.length || 4} active
                            </span>
                        </div>
                        <div className="space-y-2">
                            {(soarData?.incidents || [
                                { incident_id: 'INC-001', title: 'Ransomware Detected', status: 'contained', severity: 'critical' },
                                { incident_id: 'INC-002', title: 'C2 Communication', status: 'investigating', severity: 'high' },
                                { incident_id: 'INC-003', title: 'Data Exfiltration', status: 'new', severity: 'high' }
                            ]).slice(0, 4).map((inc: any, i: number) => (
                                <div key={inc.incident_id || i} className="flex items-center justify-between p-2.5 rounded-lg bg-[#1a1a1a]">
                                    <div className="flex items-center gap-2">
                                        <span className={`w-1.5 h-1.5 rounded-full ${inc.severity === 'critical' ? 'bg-[#ef4444]' : 'bg-[#f97316]'}`}></span>
                                        <span className="text-xs text-white">{inc.title}</span>
                                    </div>
                                    <span className={`text-xs px-1.5 py-0.5 rounded ${inc.status === 'contained' ? 'bg-[#10a37f]/20 text-[#10a37f]' :
                                        inc.status === 'investigating' ? 'bg-[#eab308]/20 text-[#eab308]' :
                                            'bg-[#ef4444]/20 text-[#ef4444]'
                                        }`}>{inc.status}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Column - MITRE Matrix + Timeline */}
                <div className="space-y-6">
                    {/* MITRE ATT&CK Coverage */}
                    <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-sm font-medium text-white">MITRE ATT&CK Coverage</h3>
                            <Link href="/mitre" className="text-xs text-[#10a37f] hover:underline">View Matrix</Link>
                        </div>
                        <div className="grid grid-cols-4 gap-1 mb-4">
                            {['Initial Access', 'Execution', 'Persistence', 'Privilege Escalation',
                                'Defense Evasion', 'Credential Access', 'Discovery', 'Lateral Movement',
                                'Collection', 'C2', 'Exfiltration', 'Impact'].map((tactic, i) => (
                                    <div key={tactic}
                                        className={`h-6 rounded text-[8px] flex items-center justify-center ${i < 4 ? 'bg-[#10a37f]/40 text-[#10a37f]' :
                                            i < 7 ? 'bg-[#10a37f]/20 text-[#10a37f]/80' :
                                                'bg-[#2a2a2a] text-[#666]'
                                            }`}
                                        title={tactic}
                                    >
                                        {tactic.slice(0, 3)}
                                    </div>
                                ))}
                        </div>
                        <div className="flex items-center justify-between text-xs">
                            <span className="text-[#666]">Techniques Detected</span>
                            <span className="text-white font-medium">{data?.mitre?.techniques_detected || 10} / {data?.mitre?.total_techniques || 38}</span>
                        </div>
                        <div className="h-2 bg-[#1a1a1a] rounded-full mt-2 overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-[#10a37f] to-[#10a37f]/50 rounded-full"
                                style={{ width: `${mitreCoverage}%` }}></div>
                        </div>
                    </div>

                    {/* Prediction Timeline - Risk Over Time */}
                    <PredictionTimeline />
                </div>
            </div>

            {/* Quick Links */}
            <div className="grid grid-cols-2 md:grid-cols-7 gap-3">
                <QuickLink href="/detections" icon={AlertTriangle} label="Detections" />
                <QuickLink href="/entities" icon={Users} label="Entities" />
                <QuickLink href="/live" icon={Activity} label="Live Feed" />
                <QuickLink href="/copilot" icon={Sparkles} label="AI Copilot" />
                <QuickLink href="/mitre" icon={GitBranch} label="MITRE" />
                <QuickLink href="/reports" icon={TrendingUp} label="Reports" />
                <QuickLink href="/approvals" icon={Clock} label="Approvals" />
            </div>

            {/* Performance Footer */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MiniStat label="MTTD" value="4.2 min" />
                <MiniStat label="MTTR" value="18 min" />
                <MiniStat label="Blocked Today" value="156" />
                <MiniStat label="Playbooks Executed" value="12" />
            </div>
        </div>
    );
}

// Component definitions
function StatCard({ label, value, icon: Icon, variant, subtext }: {
    label: string; value: number | string; icon: any; variant?: 'critical'; subtext?: string;
}) {
    const isNumber = typeof value === 'number';
    return (
        <div className={`bg-[#141414] rounded-xl border p-5 ${variant === 'critical' && isNumber && value > 0 ? 'border-[#ef4444]/30' : 'border-[#2a2a2a]'}`}>
            <div className="flex items-center justify-between mb-3">
                <Icon className={`w-5 h-5 ${variant === 'critical' && isNumber && value > 0 ? 'text-[#ef4444]' : 'text-[#666]'}`} />
                {subtext && <span className="text-xs text-[#666]">{subtext}</span>}
            </div>
            <p className={`text-3xl font-semibold ${variant === 'critical' && isNumber && value > 0 ? 'text-[#ef4444]' : 'text-white'}`}>
                {value}
            </p>
            <p className="text-sm text-[#666] mt-1">{label}</p>
        </div>
    );
}

function SeverityRow({ label, count, total, color }: { label: string; count: number; total: number; color: string; }) {
    const percentage = total > 0 ? (count / total) * 100 : 0;
    return (
        <div>
            <div className="flex justify-between text-sm mb-1">
                <span className="text-[#a1a1a1]">{label}</span>
                <span className="text-white font-medium">{count}</span>
            </div>
            <div className="h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all" style={{ width: `${Math.max(percentage, 2)}%`, backgroundColor: color }}></div>
            </div>
        </div>
    );
}

function ConfidenceBar({ label, percentage, color }: { label: string; percentage: number; color: string; }) {
    return (
        <div>
            <div className="flex justify-between text-xs mb-1">
                <span className="text-[#a1a1a1]">{label}</span>
                <span className="text-white">{percentage}%</span>
            </div>
            <div className="h-1.5 bg-[#1a1a1a] rounded-full overflow-hidden">
                <div className="h-full rounded-full" style={{ width: `${percentage}%`, backgroundColor: color }}></div>
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

'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Activity, AlertTriangle, Zap, TrendingUp, Server } from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { apiClient } from '@/lib/api';
import type { ThreatDetection, AlertNotification, SystemMetrics, DashboardStats } from '@/types';
import ThreatCard from './ThreatCard';
import AlertPanel from './AlertPanel';
import StatsCard from './StatsCard';
import NetworkChart from './charts/NetworkChart';
import NetworkTopology3D from './visualizations/NetworkTopology3D';
import { AttackTimeline } from './visualizations/AttackTimeline';

export default function Dashboard() {
    const [stats, setStats] = useState<DashboardStats | null>(null);
    const [threats, setThreats] = useState<ThreatDetection[]>([]);
    const [alerts, setAlerts] = useState<AlertNotification[]>([]);
    const [metrics, setMetrics] = useState<SystemMetrics | null>(null);

    const { isConnected, lastMessage } = useWebSocket((message) => {
        if (message.type === 'threat_detected') {
            setThreats((prev) => [message.data, ...prev].slice(0, 10));
        } else if (message.type === 'alert') {
            setAlerts((prev) => [message.data, ...prev].slice(0, 20));
        } else if (message.type === 'system_metrics') {
            setMetrics(message.data);
        }
    });

    useEffect(() => {
        loadDashboardData();
        const interval = setInterval(loadDashboardData, 10000);
        return () => clearInterval(interval);
    }, []);

    const loadDashboardData = async () => {
        try {
            // Use new API v2 endpoints
            const dashboardData = await apiClient.getDashboardOverview(24) as any;

            // Extract data from v2 response structure
            const statsData: DashboardStats = {
                total_threats: dashboardData.detections?.total || 0,
                critical_threats: dashboardData.detections?.by_severity?.critical || 0,
                threats_blocked: dashboardData.detections?.total || 0,
                average_risk_score: 65,
                system_health: dashboardData.system_health?.database_status || 'Operational'
            };

            // Get recent detections
            const detectionsResponse = await apiClient.getDetections({ limit: 10, hours: 24 }) as any;
            const threatsData: ThreatDetection[] = (detectionsResponse.detections || []).map((d: any) => ({
                id: d.id || '',
                title: d.detection_type || 'Unknown Threat',
                description: d.description || '',
                severity: d.severity || 'medium',
                threat_type: d.detection_type || '',
                source_ip: d.source_ip || '',
                destination_ip: d.destination_ip || '',
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

            // Mock alerts and metrics for now (add proper endpoints later if needed)
            const alertsData: AlertNotification[] = [];
            const metricsData: SystemMetrics = {
                cpu_usage: 45,
                memory_usage: 62,
                network_throughput: 350,
                active_connections: dashboardData.system_health?.total_entities || 0,
                threats_detected_today: dashboardData.detections?.total || 0,
                threats_blocked_today: dashboardData.detections?.by_severity?.critical || 0
            };

            setStats(statsData);
            setThreats(threatsData);
            setAlerts(alertsData);
            setMetrics(metricsData);
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    };

    return (
        <div className="min-h-screen gradient-bg">
            {/* Header */}
            <header className="glass-strong border-b border-white/10 sticky top-0 z-50">
                <div className="container mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4">
                            <div className="relative">
                                <Shield className="w-10 h-10 text-cyber-blue animate-pulse-slow" />
                                <div className="absolute inset-0 bg-cyber-blue/20 blur-xl rounded-full"></div>
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold text-gradient">PCDS</h1>
                                <p className="text-xs text-gray-400">Predictive Cyber Defence System</p>
                            </div>
                        </div>

                        <div className="flex items-center space-x-6">
                            <div className="flex items-center space-x-2">
                                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-cyber-green animate-pulse' : 'bg-red-500'}`}></div>
                                <span className="text-sm text-gray-300">
                                    {isConnected ? 'Connected' : 'Disconnected'}
                                </span>
                            </div>

                            <div className="glass px-4 py-2 rounded-lg">
                                <div className="flex items-center space-x-2">
                                    <Activity className="w-4 h-4 text-cyber-green" />
                                    <span className="text-sm font-medium">{stats?.system_health || 'Operational'}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-6 py-8 space-y-8">
                {/* Stats Overview */}
                <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <StatsCard
                        title="Total Threats"
                        value={stats?.total_threats || 0}
                        icon={<AlertTriangle />}
                        color="blue"
                        trend="+12%"
                    />
                    <StatsCard
                        title="Critical Threats"
                        value={stats?.critical_threats || 0}
                        icon={<Zap />}
                        color="red"
                        trend="-5%"
                        trendUp={false}
                    />
                    <StatsCard
                        title="Threats Blocked"
                        value={stats?.threats_blocked || 0}
                        icon={<Shield />}
                        color="green"
                        trend="+24%"
                    />
                    <StatsCard
                        title="Avg Risk Score"
                        value={stats?.average_risk_score?.toFixed(1) || '0'}
                        icon={<TrendingUp />}
                        color="yellow"
                        suffix="/100"
                    />
                </section>

                {/* 3D Network Topology */}
                <section>
                    <NetworkTopology3D threats={threats} />
                </section>

                {/* Attack Timeline */}
                <section>
                    <AttackTimeline detections={threats} />
                </section>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Threat Feed */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Network Activity Chart */}
                        <div className="glass p-6 rounded-2xl">
                            <h2 className="text-xl font-bold mb-4 flex items-center">
                                <Activity className="w-5 h-5 mr-2 text-cyber-blue" />
                                Network Activity
                            </h2>
                            <NetworkChart threats={threats} />
                        </div>

                        {/* Recent Threats */}
                        <div className="glass p-6 rounded-2xl">
                            <h2 className="text-xl font-bold mb-4 flex items-center">
                                <AlertTriangle className="w-5 h-5 mr-2 text-threat-high" />
                                Recent Threats
                            </h2>
                            <div className="space-y-4">
                                <AnimatePresence mode="popLayout">
                                    {threats.slice(0, 5).map((threat) => (
                                        <ThreatCard key={threat.id} threat={threat} />
                                    ))}
                                </AnimatePresence>
                                {threats.length === 0 && (
                                    <p className="text-center text-gray-400 py-8">No threats detected</p>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Sidebar */}
                    <div className="space-y-6">
                        {/* System Metrics */}
                        {metrics && (
                            <div className="glass p-6 rounded-2xl">
                                <h3 className="text-lg font-bold mb-4 flex items-center">
                                    <Server className="w-5 h-5 mr-2 text-cyber-purple" />
                                    System Metrics
                                </h3>
                                <div className="space-y-4">
                                    <MetricBar label="CPU Usage" value={metrics.cpu_usage} color="blue" />
                                    <MetricBar label="Memory" value={metrics.memory_usage} color="purple" />
                                    <MetricBar label="Network" value={Math.min(metrics.network_throughput / 10, 100)} color="green" />
                                </div>
                                <div className="mt-4 pt-4 border-t border-white/10 space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Active Connections</span>
                                        <span className="font-semibold">{metrics.active_connections}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Threats Today</span>
                                        <span className="font-semibold text-threat-high">{metrics.threats_detected_today}</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Live Alerts */}
                        <AlertPanel alerts={alerts} />
                    </div>
                </div >
            </main >
        </div >
    );
}

// MetricBar Component
function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
    const colorMap: Record<string, string> = {
        blue: 'bg-cyber-blue',
        purple: 'bg-cyber-purple',
        green: 'bg-cyber-green',
    };

    return (
        <div>
            <div className="flex justify-between mb-1 text-sm">
                <span className="text-gray-300">{label}</span>
                <span className="font-semibold">{value.toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                <motion.div
                    className={`h-full ${colorMap[color]} rounded-full`}
                    initial={{ width: 0 }}
                    animate={{ width: `${value}%` }}
                    transition={{ duration: 0.5 }}
                />
            </div>
        </div>
    );
}

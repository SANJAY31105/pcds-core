'use client';

import { useState } from 'react';
import { Bell, AlertTriangle, Shield, CheckCircle, Clock, X, Filter, Eye, Trash2 } from 'lucide-react';

interface Alert {
    id: string;
    title: string;
    description: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    type: 'detection' | 'system' | 'action';
    timestamp: string;
    read: boolean;
    source: string;
}

const mockAlerts: Alert[] = [
    { id: '1', title: 'Critical Detection: Ransomware Pattern', description: 'Ransomware encryption behavior detected on host-042', severity: 'critical', type: 'detection', timestamp: '2024-12-08T10:30:00Z', read: false, source: 'ML Engine' },
    { id: '2', title: 'Automated Response Triggered', description: 'Host host-042 has been isolated from network', severity: 'high', type: 'action', timestamp: '2024-12-08T10:30:05Z', read: false, source: 'Playbook Engine' },
    { id: '3', title: 'Brute Force Attack Blocked', description: '47 failed SSH attempts from 192.168.1.50', severity: 'high', type: 'detection', timestamp: '2024-12-08T09:15:00Z', read: true, source: 'Detection Engine' },
    { id: '4', title: 'New Investigation Created', description: 'Investigation INV-2024-089 created for lateral movement', severity: 'medium', type: 'system', timestamp: '2024-12-08T08:45:00Z', read: true, source: 'System' },
    { id: '5', title: 'Model Training Complete', description: 'Anomaly detection model v2.1 trained successfully', severity: 'low', type: 'system', timestamp: '2024-12-08T06:00:00Z', read: true, source: 'ML Engine' },
    { id: '6', title: 'Suspicious DNS Query', description: 'DGA-like domain pattern detected: xkcd42abc.top', severity: 'medium', type: 'detection', timestamp: '2024-12-07T22:30:00Z', read: true, source: 'DNS Monitor' },
];

export default function AlertsPage() {
    const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);
    const [filter, setFilter] = useState<'all' | 'unread' | 'critical'>('all');

    const filteredAlerts = alerts.filter(alert => {
        if (filter === 'unread') return !alert.read;
        if (filter === 'critical') return alert.severity === 'critical';
        return true;
    });

    const markAsRead = (id: string) => {
        setAlerts(alerts.map(a => a.id === id ? { ...a, read: true } : a));
    };

    const deleteAlert = (id: string) => {
        setAlerts(alerts.filter(a => a.id !== id));
    };

    const markAllRead = () => {
        setAlerts(alerts.map(a => ({ ...a, read: true })));
    };

    const getSeverityColor = (severity: string) => {
        const colors = {
            critical: 'text-red-400 bg-red-500/20 border-red-500/50',
            high: 'text-orange-400 bg-orange-500/20 border-orange-500/50',
            medium: 'text-yellow-400 bg-yellow-500/20 border-yellow-500/50',
            low: 'text-green-400 bg-green-500/20 border-green-500/50'
        };
        return colors[severity as keyof typeof colors] || colors.low;
    };

    const getTypeIcon = (type: string) => {
        if (type === 'detection') return <AlertTriangle className="w-5 h-5" />;
        if (type === 'action') return <Shield className="w-5 h-5" />;
        return <Bell className="w-5 h-5" />;
    };

    const unreadCount = alerts.filter(a => !a.read).length;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-semibold text-white flex items-center gap-3">
                        Alerts
                        {unreadCount > 0 && (
                            <span className="px-2 py-0.5 bg-red-500 text-white text-sm rounded-full">
                                {unreadCount}
                            </span>
                        )}
                    </h1>
                    <p className="text-[#666] mt-1">Security alerts and notifications</p>
                </div>
                <button
                    onClick={markAllRead}
                    className="flex items-center gap-2 px-4 py-2.5 bg-[#141414] border border-[#2a2a2a] text-[#a1a1a1] rounded-lg hover:text-white transition-colors"
                >
                    <CheckCircle className="w-4 h-4" />
                    Mark All Read
                </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Bell className="w-5 h-5 text-[#10a37f]" />
                        <span className="text-sm text-[#666]">Total Alerts</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{alerts.length}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                        <span className="text-sm text-[#666]">Critical</span>
                    </div>
                    <p className="text-3xl font-bold text-red-400">{alerts.filter(a => a.severity === 'critical').length}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Eye className="w-5 h-5 text-yellow-400" />
                        <span className="text-sm text-[#666]">Unread</span>
                    </div>
                    <p className="text-3xl font-bold text-yellow-400">{unreadCount}</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <Clock className="w-5 h-5 text-blue-400" />
                        <span className="text-sm text-[#666]">Last 24h</span>
                    </div>
                    <p className="text-3xl font-bold text-white">{alerts.filter(a => {
                        const alertTime = new Date(a.timestamp).getTime();
                        const dayAgo = Date.now() - 24 * 60 * 60 * 1000;
                        return alertTime > dayAgo;
                    }).length}</p>
                </div>
            </div>

            {/* Filters */}
            <div className="flex gap-2">
                {(['all', 'unread', 'critical'] as const).map((f) => (
                    <button
                        key={f}
                        onClick={() => setFilter(f)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors capitalize ${filter === f
                                ? 'bg-[#10a37f] text-white'
                                : 'bg-[#141414] border border-[#2a2a2a] text-[#a1a1a1] hover:text-white'
                            }`}
                    >
                        {f}
                    </button>
                ))}
            </div>

            {/* Alerts List */}
            <div className="space-y-3">
                {filteredAlerts.map((alert) => (
                    <div
                        key={alert.id}
                        className={`bg-[#141414] rounded-xl border ${alert.read ? 'border-[#2a2a2a]' : 'border-[#10a37f]/50'} p-5 transition-all hover:bg-[#1a1a1a]`}
                    >
                        <div className="flex items-start gap-4">
                            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getSeverityColor(alert.severity)}`}>
                                {getTypeIcon(alert.type)}
                            </div>
                            <div className="flex-1">
                                <div className="flex items-center gap-3 mb-1">
                                    <h3 className={`font-medium ${alert.read ? 'text-[#a1a1a1]' : 'text-white'}`}>
                                        {alert.title}
                                    </h3>
                                    {!alert.read && (
                                        <span className="w-2 h-2 bg-[#10a37f] rounded-full"></span>
                                    )}
                                    <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${getSeverityColor(alert.severity)}`}>
                                        {alert.severity}
                                    </span>
                                </div>
                                <p className="text-sm text-[#666] mb-2">{alert.description}</p>
                                <div className="flex items-center gap-4 text-xs text-[#666]">
                                    <span>{new Date(alert.timestamp).toLocaleString()}</span>
                                    <span>•</span>
                                    <span>{alert.source}</span>
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                {!alert.read && (
                                    <button
                                        onClick={() => markAsRead(alert.id)}
                                        className="p-2 hover:bg-[#2a2a2a] rounded-lg transition-colors"
                                        title="Mark as read"
                                    >
                                        <Eye className="w-4 h-4 text-[#666]" />
                                    </button>
                                )}
                                <button
                                    onClick={() => deleteAlert(alert.id)}
                                    className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
                                    title="Delete"
                                >
                                    <Trash2 className="w-4 h-4 text-[#666] hover:text-red-400" />
                                </button>
                            </div>
                        </div>
                    </div>
                ))}

                {filteredAlerts.length === 0 && (
                    <div className="text-center py-12 text-[#666]">
                        <Bell className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>No alerts to show</p>
                    </div>
                )}
            </div>
        </div>
    );
}

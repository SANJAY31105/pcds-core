'use client';

import { useState, useEffect } from 'react';

interface Incident {
    incident_id: string;
    title: string;
    description: string;
    severity: string;
    status: string;
    source: string;
    created_at: string;
    attack_type?: string;
    affected_hosts: string[];
    actions_taken: any[];
    playbook_id?: string;
}

interface Playbook {
    playbook_id: string;
    name: string;
    description: string;
    enabled: boolean;
    execution_count: number;
    trigger_conditions: any;
    action_count: number;
}

export default function SOARPage() {
    const [incidents, setIncidents] = useState<Incident[]>([]);
    const [playbooks, setPlaybooks] = useState<Playbook[]>([]);
    const [stats, setStats] = useState<any>(null);
    const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
    const [loading, setLoading] = useState(false);

    const API_BASE = 'http://localhost:8000/api/v2/soar';

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, []);

    const fetchData = async () => {
        try {
            const [incRes, pbRes, statsRes] = await Promise.all([
                fetch(`${API_BASE}/incidents`),
                fetch(`${API_BASE}/playbooks`),
                fetch(`${API_BASE}/stats`)
            ]);
            const incData = await incRes.json();
            const pbData = await pbRes.json();
            const statsData = await statsRes.json();
            setIncidents(incData.incidents || []);
            setPlaybooks(pbData.playbooks || []);
            setStats(statsData);
        } catch (error) {
            console.error('Failed to fetch SOAR data:', error);
        }
    };

    const simulateAttack = async (attackType: string) => {
        setLoading(true);
        try {
            await fetch(`${API_BASE}/simulate/ml-detection?attack_type=${attackType}`, { method: 'POST' });
            await fetchData();
        } catch (error) {
            console.error('Failed to simulate:', error);
        }
        setLoading(false);
    };

    const togglePlaybook = async (playbookId: string, enabled: boolean) => {
        try {
            await fetch(`${API_BASE}/playbooks/${playbookId}/toggle?enabled=${enabled}`, { method: 'POST' });
            await fetchData();
        } catch (error) {
            console.error('Failed to toggle playbook:', error);
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity.toLowerCase()) {
            case 'critical': return 'bg-[#ef4444]';
            case 'high': return 'bg-[#f97316]';
            case 'medium': return 'bg-[#eab308]';
            case 'low': return 'bg-[#3b82f6]';
            default: return 'bg-[#666]';
        }
    };

    const getStatusColor = (status: string) => {
        switch (status.toLowerCase()) {
            case 'new': return 'text-[#ef4444]';
            case 'triaged': return 'text-[#eab308]';
            case 'investigating': return 'text-[#3b82f6]';
            case 'contained': return 'text-[#22c55e]';
            case 'closed': return 'text-[#666]';
            default: return 'text-white';
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white flex items-center gap-3">
                    🎯 SOAR - Security Orchestration
                </h1>
                <p className="text-[#666] text-sm mt-1">
                    Automated incident response and playbook orchestration
                </p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Total Incidents</div>
                    <div className="text-3xl font-bold text-white">{stats?.total_incidents || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Auto Triaged</div>
                    <div className="text-3xl font-bold text-[#22c55e]">{stats?.incidents_auto_triaged || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Playbooks Run</div>
                    <div className="text-3xl font-bold text-[#3b82f6]">{stats?.playbooks_executed || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Actions Executed</div>
                    <div className="text-3xl font-bold text-[#a855f7]">{stats?.actions_executed || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Active Playbooks</div>
                    <div className="text-3xl font-bold text-[#eab308]">{stats?.total_playbooks || 0}</div>
                </div>
            </div>

            {/* Simulate Attack */}
            <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                <h3 className="text-white font-medium mb-3">🧪 Simulate Attack (Test Playbooks)</h3>
                <div className="flex flex-wrap gap-2">
                    {['DDoS', 'Ransomware', 'SQL Injection', 'SSH-Patator', 'PortScan'].map(attack => (
                        <button
                            key={attack}
                            onClick={() => simulateAttack(attack)}
                            disabled={loading}
                            className="bg-[#1a1a1a] hover:bg-[#252525] text-white px-4 py-2 rounded-lg transition disabled:opacity-50 border border-[#2a2a2a]"
                        >
                            {attack}
                        </button>
                    ))}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Incidents List */}
                <div className="lg:col-span-2 bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h2 className="text-lg font-medium text-white mb-4">🚨 Security Incidents</h2>
                    {incidents.length === 0 ? (
                        <div className="text-center text-[#666] py-12">
                            <div className="text-6xl mb-4">✅</div>
                            <p>No active incidents</p>
                            <p className="text-sm mt-2">Simulate an attack to test the SOAR system</p>
                        </div>
                    ) : (
                        <div className="space-y-3 max-h-[500px] overflow-y-auto">
                            {incidents.map(incident => (
                                <div
                                    key={incident.incident_id}
                                    onClick={() => setSelectedIncident(incident)}
                                    className={`bg-[#0a0a0a] rounded-lg p-4 cursor-pointer hover:bg-[#1a1a1a] transition border-l-4 ${getSeverityColor(incident.severity)}`}
                                >
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <span className="text-white font-medium">{incident.title}</span>
                                                <span className={`text-xs px-2 py-0.5 rounded ${getSeverityColor(incident.severity)}`}>
                                                    {incident.severity.toUpperCase()}
                                                </span>
                                            </div>
                                            <div className="text-[#666] text-sm mt-1">
                                                {incident.incident_id} • {incident.source}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className={`text-sm font-medium ${getStatusColor(incident.status)}`}>
                                                {incident.status}
                                            </div>
                                            <div className="text-xs text-[#666]">
                                                {new Date(incident.created_at).toLocaleTimeString()}
                                            </div>
                                        </div>
                                    </div>
                                    {incident.playbook_id && (
                                        <div className="mt-2 text-xs text-[#3b82f6]">▶ Playbook: {incident.playbook_id}</div>
                                    )}
                                    {incident.actions_taken.length > 0 && (
                                        <div className="mt-2 text-xs text-[#22c55e]">✓ {incident.actions_taken.length} actions taken</div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Playbooks */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h2 className="text-lg font-medium text-white mb-4">📋 Response Playbooks</h2>
                    <div className="space-y-3">
                        {playbooks.map(pb => (
                            <div key={pb.playbook_id} className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <div className="text-white font-medium">{pb.name}</div>
                                        <div className="text-[#666] text-xs mt-1">{pb.description}</div>
                                    </div>
                                    <button
                                        onClick={() => togglePlaybook(pb.playbook_id, !pb.enabled)}
                                        className={`text-xs px-2 py-1 rounded ${pb.enabled ? 'bg-[#22c55e]/20 text-[#22c55e]' : 'bg-[#666]/20 text-[#666]'}`}
                                    >
                                        {pb.enabled ? 'Enabled' : 'Disabled'}
                                    </button>
                                </div>
                                <div className="flex gap-4 mt-3 text-xs">
                                    <span className="text-[#666]">📦 {pb.action_count} actions</span>
                                    <span className="text-[#666]">▶ {pb.execution_count} runs</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Incident Detail Modal */}
            {selectedIncident && (
                <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
                    <div className="bg-[#141414] rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-[#2a2a2a]">
                        <div className="p-6">
                            <div className="flex justify-between items-start mb-4">
                                <div>
                                    <h2 className="text-2xl font-bold text-white">{selectedIncident.title}</h2>
                                    <div className="text-[#666]">{selectedIncident.incident_id}</div>
                                </div>
                                <button onClick={() => setSelectedIncident(null)} className="text-[#666] hover:text-white text-2xl">×</button>
                            </div>
                            <div className="grid grid-cols-2 gap-4 mb-4">
                                <div>
                                    <span className="text-[#666] text-sm">Severity</span>
                                    <div className={`${getSeverityColor(selectedIncident.severity)} text-white px-2 py-1 rounded inline-block mt-1`}>
                                        {selectedIncident.severity.toUpperCase()}
                                    </div>
                                </div>
                                <div>
                                    <span className="text-[#666] text-sm">Status</span>
                                    <div className={`${getStatusColor(selectedIncident.status)} font-medium mt-1`}>{selectedIncident.status}</div>
                                </div>
                            </div>
                            <div className="mb-4">
                                <span className="text-[#666] text-sm">Description</span>
                                <p className="text-white mt-1">{selectedIncident.description}</p>
                            </div>
                            {selectedIncident.attack_type && (
                                <div className="mb-4">
                                    <span className="text-[#666] text-sm">Attack Type</span>
                                    <p className="text-[#ef4444] font-medium mt-1">{selectedIncident.attack_type}</p>
                                </div>
                            )}
                            {selectedIncident.affected_hosts.length > 0 && (
                                <div className="mb-4">
                                    <span className="text-[#666] text-sm">Affected Hosts</span>
                                    <div className="flex flex-wrap gap-2 mt-1">
                                        {selectedIncident.affected_hosts.map((host, i) => (
                                            <span key={i} className="bg-[#1a1a1a] text-white px-2 py-1 rounded text-sm">{host}</span>
                                        ))}
                                    </div>
                                </div>
                            )}
                            {selectedIncident.actions_taken.length > 0 && (
                                <div>
                                    <span className="text-[#666] text-sm">Actions Taken</span>
                                    <div className="space-y-2 mt-2">
                                        {selectedIncident.actions_taken.map((action, i) => (
                                            <div key={i} className="bg-[#0a0a0a] rounded p-3 text-sm border border-[#2a2a2a]">
                                                <div className="flex justify-between">
                                                    <span className="text-white font-medium">{action.action_type}</span>
                                                    <span className={action.status === 'completed' ? 'text-[#22c55e]' : 'text-[#eab308]'}>{action.status}</span>
                                                </div>
                                                <div className="text-[#666] text-xs mt-1">Target: {action.target}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

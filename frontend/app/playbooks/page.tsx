'use client';

import { useState, useEffect } from 'react';

interface Playbook {
    id: string;
    name: string;
    description: string;
    trigger_conditions: {
        technique_ids?: string[];
        severity?: string[];
        detection_types?: string[];
    };
    action_count: number;
}

interface Execution {
    id: string;
    playbook_id: string;
    status: string;
    started_at: string;
    completed_at: string;
    actions_completed: number;
    actions_failed: number;
}

export default function PlaybooksPage() {
    const [playbooks, setPlaybooks] = useState<Playbook[]>([]);
    const [executions, setExecutions] = useState<Execution[]>([]);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState<'playbooks' | 'executions'>('playbooks');

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            const [pbRes, exRes] = await Promise.all([
                fetch('/api/v2/playbooks/'),
                fetch('/api/v2/playbooks/executions/history')
            ]);

            const pbData = await pbRes.json();
            const exData = await exRes.json();

            setPlaybooks(pbData.playbooks || []);
            setExecutions(exData.executions || []);
        } catch (error) {
            console.error('Failed to fetch playbooks:', error);
        } finally {
            setLoading(false);
        }
    };

    const triggerPlaybook = async (playbookId: string) => {
        try {
            const response = await fetch(`/api/v2/playbooks/execute/${playbookId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    entity_id: 'test-entity',
                    detection_type: 'manual_trigger',
                    severity: 'high'
                })
            });

            if (response.ok) {
                fetchData(); // Refresh
            }
        } catch (error) {
            console.error('Failed to trigger playbook:', error);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'bg-green-500/20 text-green-400 border-green-500/30';
            case 'running': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
            case 'failed': return 'bg-red-500/20 text-red-400 border-red-500/30';
            default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
        }
    };

    const getPlaybookIcon = (id: string) => {
        const icons: Record<string, string> = {
            'ransomware_response': '🔐',
            'credential_theft': '🔑',
            'lateral_movement': '🚨',
            'data_exfiltration': '📤',
            'c2_communication': '📡',
            'insider_threat': '👤',
            'malware_detection': '🦠'
        };
        return icons[id] || '🛡️';
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500"></div>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white">Automated Playbooks</h1>
                    <p className="text-gray-400">Response automation and threat containment</p>
                </div>
                <div className="flex items-center gap-2 bg-gradient-to-r from-green-500/20 to-emerald-500/20 px-4 py-2 rounded-lg border border-green-500/30">
                    <span className="text-2xl">🤖</span>
                    <div>
                        <div className="text-green-400 font-semibold">{playbooks.length} Playbooks</div>
                        <div className="text-xs text-gray-400">Active & Ready</div>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2">
                <button
                    onClick={() => setActiveTab('playbooks')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'playbooks'
                            ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                            : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                        }`}
                >
                    Playbooks
                </button>
                <button
                    onClick={() => setActiveTab('executions')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${activeTab === 'executions'
                            ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                            : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
                        }`}
                >
                    Execution History
                </button>
            </div>

            {/* Playbooks Grid */}
            {activeTab === 'playbooks' && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {playbooks.map((playbook) => (
                        <div
                            key={playbook.id}
                            className="bg-gradient-to-br from-gray-900/80 to-gray-800/50 rounded-xl border border-gray-700/50 p-5 hover:border-cyan-500/30 transition-all"
                        >
                            <div className="flex items-start justify-between mb-3">
                                <span className="text-3xl">{getPlaybookIcon(playbook.id)}</span>
                                <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-1 rounded-full border border-purple-500/30">
                                    {playbook.action_count} actions
                                </span>
                            </div>

                            <h3 className="text-lg font-semibold text-white mb-2">{playbook.name}</h3>
                            <p className="text-sm text-gray-400 mb-4">{playbook.description}</p>

                            {/* Triggers */}
                            <div className="mb-4">
                                <div className="text-xs text-gray-500 mb-1">Triggers:</div>
                                <div className="flex flex-wrap gap-1">
                                    {playbook.trigger_conditions.technique_ids?.slice(0, 3).map((t) => (
                                        <span key={t} className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">
                                            {t}
                                        </span>
                                    ))}
                                    {playbook.trigger_conditions.severity?.map((s) => (
                                        <span key={s} className="text-xs bg-orange-500/20 text-orange-400 px-2 py-0.5 rounded">
                                            {s}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            <button
                                onClick={() => triggerPlaybook(playbook.id)}
                                className="w-full py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-all border border-cyan-500/30 font-medium"
                            >
                                ▶ Execute Manually
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Execution History */}
            {activeTab === 'executions' && (
                <div className="bg-gray-900/50 rounded-xl border border-gray-700/50 overflow-hidden">
                    <table className="w-full">
                        <thead className="bg-gray-800/50">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">Playbook</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">Status</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">Started</th>
                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-700/50">
                            {executions.length === 0 ? (
                                <tr>
                                    <td colSpan={4} className="px-4 py-8 text-center text-gray-500">
                                        No playbook executions yet
                                    </td>
                                </tr>
                            ) : (
                                executions.map((exec) => (
                                    <tr key={exec.id} className="hover:bg-gray-800/30">
                                        <td className="px-4 py-3 text-white font-medium">{exec.playbook_id}</td>
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-1 rounded text-xs border ${getStatusColor(exec.status)}`}>
                                                {exec.status}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-gray-400 text-sm">
                                            {exec.started_at ? new Date(exec.started_at).toLocaleString() : '-'}
                                        </td>
                                        <td className="px-4 py-3 text-sm">
                                            <span className="text-green-400">{exec.actions_completed}</span>
                                            {exec.actions_failed > 0 && (
                                                <span className="text-red-400 ml-2">({exec.actions_failed} failed)</span>
                                            )}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

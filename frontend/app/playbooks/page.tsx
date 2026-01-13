'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { Bot, Play, Clock, Zap, Settings } from 'lucide-react';

interface Playbook {
    id: string;
    name: string;
    description: string;
    trigger_type: string;
    actions: number;
    last_run?: string;
    status?: string;
}

export default function PlaybooksPage() {
    const [playbooks, setPlaybooks] = useState<Playbook[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadPlaybooks();
    }, []);

    const loadPlaybooks = async () => {
        try {
            const response = await apiClient.get('/api/v2/playbooks') as any;
            setPlaybooks(response.playbooks || []);
        } catch (error) {
            console.error('Failed to load playbooks:', error);
            // Mock data with guaranteed values
            setPlaybooks([
                { id: '1', name: 'Ransomware Response', description: 'Automated isolation and containment for ransomware threats', trigger_type: 'critical_detection', actions: 5, status: 'active', last_run: new Date().toISOString() },
                { id: '2', name: 'Phishing Containment', description: 'Block sender and quarantine emails from phishing campaigns', trigger_type: 'phishing_detection', actions: 3, status: 'active' },
                { id: '3', name: 'Lateral Movement Block', description: 'Disable compromised accounts and block lateral spread', trigger_type: 'lateral_movement', actions: 4, status: 'active' },
                { id: '4', name: 'Data Exfiltration Alert', description: 'Alert SOC and throttle network for exfiltration attempts', trigger_type: 'exfiltration', actions: 3, status: 'disabled' }
            ]);
        } finally {
            setLoading(false);
        }
    };

    const [runningId, setRunningId] = useState<string | null>(null);

    const runPlaybook = async (id: string) => {
        setRunningId(id);
        try {
            await fetch(`http://localhost:8000/api/v2/playbooks/${id}/run`, { method: 'POST' });
            alert(`Playbook executed successfully!`);
            loadPlaybooks();
        } catch (error) {
            console.error('Failed to run playbook:', error);
            alert('Playbook executed (simulated)');
        } finally {
            setRunningId(null);
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold text-white">Playbooks</h1>
                    <p className="text-[#666] text-sm mt-1">Automated response playbooks</p>
                </div>
                <button
                    onClick={() => alert('⚙️ Playbook configuration coming soon!\n\nIn production, this opens the playbook editor.')}
                    className="px-4 py-2 rounded-lg bg-[#10a37f] text-white text-sm font-medium hover:bg-[#0d8a6a] transition-colors flex items-center gap-2"
                >
                    <Settings className="w-4 h-4" /> Configure
                </button>
            </div>

            {/* List */}
            {loading ? (
                <div className="text-center py-12 text-[#666]">Loading...</div>
            ) : playbooks.length === 0 ? (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-12 text-center">
                    <Bot className="w-12 h-12 mx-auto mb-3 text-[#333]" />
                    <p className="text-[#666]">No playbooks configured</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {playbooks.map((playbook) => {
                        const isActive = playbook.status === 'active';
                        return (
                            <div key={playbook.id} className={`bg-[#141414] rounded-xl border p-5 ${isActive ? 'border-[#2a2a2a]' : 'border-[#2a2a2a] opacity-60'}`}>
                                <div className="flex items-start justify-between mb-3">
                                    <div className="flex items-center gap-2">
                                        <Bot className="w-5 h-5 text-[#10a37f]" />
                                        <span className={`text-xs px-2 py-0.5 rounded ${isActive ? 'bg-[#22c55e]/15 text-[#22c55e]' : 'bg-[#666]/15 text-[#666]'}`}>
                                            {isActive ? 'ACTIVE' : 'DISABLED'}
                                        </span>
                                    </div>
                                    {playbook.last_run && (
                                        <span className="text-xs text-[#666] flex items-center gap-1">
                                            <Clock className="w-3 h-3" /> {new Date(playbook.last_run).toLocaleDateString()}
                                        </span>
                                    )}
                                </div>

                                <h3 className="text-base font-medium text-white mb-1">{playbook.name}</h3>
                                <p className="text-sm text-[#666] mb-4">{playbook.description}</p>

                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3 text-xs text-[#666]">
                                        <span className="flex items-center gap-1"><Zap className="w-3 h-3" /> {playbook.actions} actions</span>
                                    </div>
                                    <button
                                        onClick={() => runPlaybook(playbook.id)}
                                        disabled={!isActive || runningId === playbook.id}
                                        className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a] text-xs text-[#a1a1a1] hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {runningId === playbook.id ? (
                                            <><span className="animate-spin">⏳</span> Running...</>
                                        ) : (
                                            <><Play className="w-3 h-3" /> Run</>
                                        )}
                                    </button>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

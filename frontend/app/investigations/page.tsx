'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { ClipboardList, Search, ChevronRight, AlertTriangle, User } from 'lucide-react';

interface Investigation {
    id: string;
    title: string;
    status: 'open' | 'in_progress' | 'closed';
    priority: string;
    assignee: string;
    created_at: string;
    entity_count: number;
    detection_count: number;
}

export default function InvestigationsPage() {
    const [investigations, setInvestigations] = useState<Investigation[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadInvestigations();
    }, []);

    const loadInvestigations = async () => {
        try {
            const response = await apiClient.get('/api/v2/investigations') as any;
            setInvestigations(response.investigations || []);
        } catch (error) {
            console.error('Failed to load investigations:', error);
            // Mock data
            setInvestigations([
                { id: '1', title: 'Ransomware Attack Investigation', status: 'in_progress', priority: 'critical', assignee: 'analyst@pcds.com', created_at: new Date().toISOString(), entity_count: 5, detection_count: 12 },
                { id: '2', title: 'Suspicious Lateral Movement', status: 'open', priority: 'high', assignee: 'unassigned', created_at: new Date().toISOString(), entity_count: 3, detection_count: 8 },
                { id: '3', title: 'Credential Theft Review', status: 'closed', priority: 'medium', assignee: 'admin@pcds.com', created_at: new Date().toISOString(), entity_count: 2, detection_count: 4 }
            ]);
        } finally {
            setLoading(false);
        }
    };

    const getStatusColor = (status: string) => {
        const colors: Record<string, string> = { open: '#eab308', in_progress: '#3b82f6', closed: '#22c55e' };
        return colors[status] || colors.open;
    };

    const getPriorityColor = (priority: string) => {
        const colors: Record<string, string> = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#3b82f6' };
        return colors[priority] || colors.medium;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold text-white">Investigations</h1>
                    <p className="text-[#666] text-sm mt-1">Active security investigations</p>
                </div>
                <button className="px-4 py-2 rounded-lg bg-[#10a37f] text-white text-sm font-medium hover:bg-[#0d8a6a] transition-colors">
                    + New Investigation
                </button>
            </div>

            {/* List */}
            {loading ? (
                <div className="text-center py-12 text-[#666]">Loading...</div>
            ) : investigations.length === 0 ? (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-12 text-center">
                    <ClipboardList className="w-12 h-12 mx-auto mb-3 text-[#333]" />
                    <p className="text-[#666]">No active investigations</p>
                </div>
            ) : (
                <div className="space-y-2">
                    {investigations.map((inv) => (
                        <div key={inv.id} className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-4 hover:bg-[#1a1a1a] transition-colors cursor-pointer">
                            <div className="flex items-center justify-between">
                                <div className="flex items-start gap-4">
                                    <AlertTriangle className="w-5 h-5 mt-0.5" style={{ color: getPriorityColor(inv.priority) }} />
                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <h3 className="text-sm font-medium text-white">{inv.title}</h3>
                                            <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: `${getStatusColor(inv.status)}20`, color: getStatusColor(inv.status) }}>
                                                {inv.status.replace('_', ' ').toUpperCase()}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-4 text-xs text-[#666]">
                                            <span className="flex items-center gap-1"><User className="w-3 h-3" /> {inv.assignee}</span>
                                            <span>{inv.entity_count} entities</span>
                                            <span>{inv.detection_count} detections</span>
                                        </div>
                                    </div>
                                </div>
                                <ChevronRight className="w-4 h-4 text-[#444]" />
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { Clock, Check, X, AlertTriangle, Shield } from 'lucide-react';

interface PendingAction {
    id: string;
    action_type: string;
    target: string;
    reason: string;
    severity: string;
    created_at: string;
    expires_at: string;
}

export default function ApprovalsPage() {
    const [pending, setPending] = useState<PendingAction[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadPending();
        const interval = setInterval(loadPending, 10000);
        return () => clearInterval(interval);
    }, []);

    const loadPending = async () => {
        try {
            const response = await apiClient.get('/api/v2/response/pending') as any;
            setPending(response.pending_actions || []);
        } catch (error) {
            console.error('Failed to load pending actions:', error);
            setPending([]);
        } finally {
            setLoading(false);
        }
    };

    const handleApprove = async (actionId: string) => {
        try {
            await fetch(`http://localhost:8000/api/v2/response/approve/${actionId}`, { method: 'POST' });
            loadPending();
        } catch (error) {
            console.error('Failed to approve:', error);
        }
    };

    const handleReject = async (actionId: string) => {
        try {
            await fetch(`http://localhost:8000/api/v2/response/reject/${actionId}`, { method: 'POST' });
            loadPending();
        } catch (error) {
            console.error('Failed to reject:', error);
        }
    };

    const getSeverityColor = (severity: string) => {
        const colors: Record<string, string> = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#3b82f6' };
        return colors[severity] || colors.medium;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold text-white">Approvals</h1>
                    <p className="text-[#666] text-sm mt-1">Pending security actions requiring approval</p>
                </div>
                <div className="px-3 py-1.5 bg-[#141414] border border-[#2a2a2a] rounded-lg">
                    <span className="text-sm text-[#a1a1a1]">{pending.length} pending</span>
                </div>
            </div>

            {/* Pending Actions */}
            {loading ? (
                <div className="text-center py-12 text-[#666]">Loading...</div>
            ) : pending.length === 0 ? (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-12 text-center">
                    <Shield className="w-12 h-12 mx-auto mb-3 text-[#22c55e]/50" />
                    <h3 className="text-lg font-medium text-white mb-1">All Clear</h3>
                    <p className="text-sm text-[#666]">No pending actions require approval</p>
                </div>
            ) : (
                <div className="space-y-3">
                    {pending.map((action) => (
                        <div key={action.id} className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <div className="flex items-start justify-between mb-4">
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <AlertTriangle className="w-4 h-4" style={{ color: getSeverityColor(action.severity) }} />
                                        <h3 className="text-sm font-medium text-white">{action.action_type}</h3>
                                        <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: `${getSeverityColor(action.severity)}20`, color: getSeverityColor(action.severity) }}>
                                            {action.severity.toUpperCase()}
                                        </span>
                                    </div>
                                    <p className="text-sm text-[#a1a1a1]">Target: {action.target}</p>
                                    <p className="text-xs text-[#666] mt-1">{action.reason}</p>
                                </div>
                                <div className="text-right text-xs text-[#666]">
                                    <div className="flex items-center gap-1">
                                        <Clock className="w-3 h-3" />
                                        {new Date(action.created_at).toLocaleTimeString()}
                                    </div>
                                </div>
                            </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => handleApprove(action.id)}
                                    className="flex-1 py-2 rounded-lg bg-[#10a37f] text-white text-sm font-medium hover:bg-[#0d8a6a] transition-colors flex items-center justify-center gap-2"
                                >
                                    <Check className="w-4 h-4" /> Approve
                                </button>
                                <button
                                    onClick={() => handleReject(action.id)}
                                    className="flex-1 py-2 rounded-lg bg-[#1a1a1a] border border-[#2a2a2a] text-[#a1a1a1] text-sm font-medium hover:text-white transition-colors flex items-center justify-center gap-2"
                                >
                                    <X className="w-4 h-4" /> Reject
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

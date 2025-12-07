'use client';

import { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Clock, AlertTriangle, Shield, RefreshCw } from 'lucide-react';

interface ApprovalRequest {
    id: string;
    detection_id: string;
    entity_id: string;
    action: string;
    confidence: number;
    impact: string;
    created_at: string;
    reason: string;
}

export default function ApprovalsPage() {
    const [approvals, setApprovals] = useState<ApprovalRequest[]>([]);
    const [loading, setLoading] = useState(true);
    const [processing, setProcessing] = useState<string | null>(null);

    useEffect(() => {
        fetchApprovals();
        const interval = setInterval(fetchApprovals, 5000); // Auto-refresh every 5s
        return () => clearInterval(interval);
    }, []);

    const fetchApprovals = async () => {
        try {
            const res = await fetch('/api/v2/response/approvals');
            const data = await res.json();
            setApprovals(data.pending || []);
        } catch (error) {
            console.error('Failed to fetch approvals:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleApprove = async (id: string) => {
        setProcessing(id);
        try {
            await fetch(`/api/v2/response/approvals/${id}/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ analyst: 'admin', notes: 'Approved from dashboard' })
            });
            fetchApprovals();
        } catch (error) {
            console.error('Failed to approve:', error);
        } finally {
            setProcessing(null);
        }
    };

    const handleReject = async (id: string) => {
        setProcessing(id);
        try {
            await fetch(`/api/v2/response/approvals/${id}/reject`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ analyst: 'admin', reason: 'Rejected from dashboard' })
            });
            fetchApprovals();
        } catch (error) {
            console.error('Failed to reject:', error);
        } finally {
            setProcessing(null);
        }
    };

    const getActionIcon = (action: string) => {
        switch (action) {
            case 'isolate_host': return '🔒';
            case 'block_ip': return '🚫';
            case 'disable_user': return '👤';
            case 'kill_process': return '💀';
            default: return '⚡';
        }
    };

    const getImpactColor = (impact: string) => {
        switch (impact) {
            case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
            case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
            case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
            default: return 'bg-green-500/20 text-green-400 border-green-500/30';
        }
    };

    const formatTime = (isoString: string) => {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        return `${Math.floor(diffMins / 60)}h ago`;
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
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <Shield className="w-8 h-8 text-yellow-400" />
                        Pending Approvals
                    </h1>
                    <p className="text-gray-400">Review and approve security response actions</p>
                </div>
                <div className="flex items-center gap-4">
                    <button
                        onClick={fetchApprovals}
                        className="flex items-center gap-2 px-4 py-2 bg-gray-700/50 rounded-lg hover:bg-gray-600/50 transition-all"
                    >
                        <RefreshCw className="w-4 h-4" />
                        Refresh
                    </button>
                    <div className={`px-4 py-2 rounded-lg border ${approvals.length > 0 ? 'bg-yellow-500/20 border-yellow-500/30' : 'bg-green-500/20 border-green-500/30'}`}>
                        <span className={approvals.length > 0 ? 'text-yellow-400' : 'text-green-400'}>
                            {approvals.length} Pending
                        </span>
                    </div>
                </div>
            </div>

            {/* Empty State */}
            {approvals.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 bg-gray-900/50 rounded-xl border border-gray-700/50">
                    <CheckCircle className="w-16 h-16 text-green-400 mb-4" />
                    <h3 className="text-xl font-semibold text-white mb-2">All Clear!</h3>
                    <p className="text-gray-400">No pending approvals at this time</p>
                </div>
            )}

            {/* Approval Cards */}
            <div className="space-y-4">
                {approvals.map((approval) => (
                    <div
                        key={approval.id}
                        className="bg-gradient-to-r from-gray-900/80 to-gray-800/50 rounded-xl border border-yellow-500/30 p-6 hover:border-yellow-400/50 transition-all"
                    >
                        <div className="flex items-start justify-between">
                            {/* Left: Info */}
                            <div className="flex items-start gap-4">
                                <div className="text-4xl">{getActionIcon(approval.action)}</div>
                                <div>
                                    <div className="flex items-center gap-3 mb-2">
                                        <h3 className="text-lg font-semibold text-white">
                                            {approval.action.replace('_', ' ').toUpperCase()}
                                        </h3>
                                        <span className={`text-xs px-2 py-1 rounded border ${getImpactColor(approval.impact)}`}>
                                            {approval.impact.toUpperCase()} IMPACT
                                        </span>
                                    </div>

                                    <div className="space-y-1 text-sm">
                                        <p className="text-gray-300">
                                            <span className="text-gray-500">Target:</span> {approval.entity_id}
                                        </p>
                                        <p className="text-gray-300">
                                            <span className="text-gray-500">Confidence:</span>
                                            <span className={approval.confidence >= 0.9 ? 'text-red-400' : 'text-yellow-400'}>
                                                {' '}{(approval.confidence * 100).toFixed(0)}%
                                            </span>
                                        </p>
                                        <p className="text-gray-400 text-xs">
                                            {approval.reason}
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Right: Actions & Time */}
                            <div className="flex flex-col items-end gap-3">
                                <div className="flex items-center gap-2 text-gray-400 text-sm">
                                    <Clock className="w-4 h-4" />
                                    {formatTime(approval.created_at)}
                                </div>

                                <div className="flex gap-2">
                                    <button
                                        onClick={() => handleReject(approval.id)}
                                        disabled={processing === approval.id}
                                        className="flex items-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 border border-red-500/30 rounded-lg hover:bg-red-500/30 transition-all disabled:opacity-50"
                                    >
                                        <XCircle className="w-4 h-4" />
                                        Reject
                                    </button>
                                    <button
                                        onClick={() => handleApprove(approval.id)}
                                        disabled={processing === approval.id}
                                        className="flex items-center gap-2 px-4 py-2 bg-green-500/20 text-green-400 border border-green-500/30 rounded-lg hover:bg-green-500/30 transition-all disabled:opacity-50"
                                    >
                                        <CheckCircle className="w-4 h-4" />
                                        Approve
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { Investigation } from '@/types';
import { FileText, Plus, Clock, User } from 'lucide-react';

export default function InvestigationsPage() {
    const [investigations, setInvestigations] = useState<Investigation[]>([]);
    const [loading, setLoading] = useState(true);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [newTitle, setNewTitle] = useState('');

    useEffect(() => {
        loadInvestigations();
    }, []);

    const loadInvestigations = async () => {
        try {
            const data = await apiClient.getInvestigations();
            setInvestigations(data.investigations || []);
        } catch (error) {
            console.error('Failed to load investigations:', error);
        } finally {
            setLoading(false);
        }
    };

    const createInvestigation = async () => {
        if (!newTitle.trim()) return;

        try {
            await apiClient.createInvestigation(newTitle);
            setNewTitle('');
            setShowCreateModal(false);
            loadInvestigations();
        } catch (error) {
            console.error('Failed to create investigation:', error);
        }
    };

    const getStatusColor = (status: string) => {
        const colors = {
            open: 'bg-green-500/20 text-green-400 border-green-500/50',
            investigating: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
            closed: 'bg-slate-500/20 text-slate-400 border-slate-500/50'
        };
        return colors[status as keyof typeof colors] || colors.open;
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                        Investigations
                    </h1>
                    <p className="text-slate-400 mt-1">Manage security investigations and case workflows</p>
                </div>
                <button
                    onClick={() => setShowCreateModal(true)}
                    className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all flex items-center space-x-2"
                >
                    <Plus className="w-5 h-5" />
                    <span>New Investigation</span>
                </button>
            </div>

            {/* Investigations Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {loading ? (
                    <div className="col-span-full text-center py-12 text-slate-400">
                        Loading investigations...
                    </div>
                ) : investigations.length === 0 ? (
                    <div className="col-span-full text-center py-12">
                        <FileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                        <p className="text-slate-400">No investigations yet</p>
                        <p className="text-slate-500 text-sm mt-2">Create your first investigation to get started</p>
                    </div>
                ) : (
                    investigations.map((inv) => (
                        <div
                            key={inv.id}
                            className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 p-6 shadow-xl hover:shadow-2xl hover:border-cyan-500/40 transition-all cursor-pointer"
                        >
                            <div className="flex items-start justify-between mb-4">
                                <FileText className="w-8 h-8 text-cyan-400" />
                                <span className={`px-3 py-1 text-xs font-bold rounded-full border ${getStatusColor(inv.status)}`}>
                                    {inv.status.toUpperCase()}
                                </span>
                            </div>

                            <h3 className="text-lg font-bold text-white mb-2">{inv.title}</h3>

                            <div className="space-y-2">
                                <div className="flex items-center text-sm text-slate-400">
                                    <User className="w-4 h-4 mr-2" />
                                    {inv.assignee}
                                </div>
                                <div className="flex items-center text-sm text-slate-400">
                                    <Clock className="w-4 h-4 mr-2" />
                                    {new Date(inv.created_at).toLocaleDateString()}
                                </div>
                            </div>

                            <div className="mt-4 pt-4 border-t border-slate-700">
                                <div className="flex items-center justify-between text-xs text-slate-500">
                                    <span>{inv.notes?.length || 0} notes</span>
                                    <span>{inv.evidence?.length || 0} evidence</span>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Create Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
                    <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl border border-cyan-500/20 p-8 max-w-md w-full mx-4">
                        <h2 className="text-2xl font-bold text-white mb-6">Create Investigation</h2>
                        <input
                            type="text"
                            placeholder="Investigation title..."
                            value={newTitle}
                            onChange={(e) => setNewTitle(e.target.value)}
                            className="w-full px-4 py-3 bg-slate-800/50 border border-cyan-500/20 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 mb-6"
                            autoFocus
                        />
                        <div className="flex items-center space-x-4">
                            <button
                                onClick={createInvestigation}
                                className="flex-1 px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all"
                            >
                                Create
                            </button>
                            <button
                                onClick={() => {
                                    setShowCreateModal(false);
                                    setNewTitle('');
                                }}
                                className="flex-1 px-4 py-3 bg-slate-700/50 text-slate-300 font-medium rounded-lg hover:bg-slate-700 transition-all"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

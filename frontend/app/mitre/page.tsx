'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { MITRETactic } from '@/types';
import { Shield, Target, Info } from 'lucide-react';

export default function MITREPage() {
    const [tactics, setTactics] = useState<MITRETactic[]>([]);
    const [heatmap, setHeatmap] = useState<Record<string, number>>({});
    const [selectedTactic, setSelectedTactic] = useState<string | null>(null);
    const [techniques, setTechniques] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const [tacticsData, heatmapData] = await Promise.all([
                apiClient.getMITRETactics() as Promise<any>,
                apiClient.getMITREHeatmap() as Promise<any>
            ]);
            setTactics(tacticsData.tactics || []);
            setHeatmap(heatmapData.heatmap || {});
        } catch (error) {
            console.error('Failed to load MITRE data:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadTechniques = async (tacticId: string) => {
        setSelectedTactic(tacticId);
        try {
            const data = await apiClient.getMITRETechnique(tacticId) as any;
            setTechniques(data.techniques || []);
        } catch (error) {
            console.error('Failed to load techniques:', error);
        }
    };

    const getHeatmapIntensity = (techniqueId: string) => {
        const count = heatmap[techniqueId] || 0;
        if (count === 0) return 'bg-slate-800/50';
        if (count <= 2) return 'bg-blue-500/30';
        if (count <= 5) return 'bg-yellow-500/40';
        if (count <= 10) return 'bg-orange-500/50';
        return 'bg-red-500/60';
    };

    const getTacticColor = (tacticId: string) => {
        const colors: Record<string, string> = {
            'TA0001': 'from-purple-500 to-pink-500',
            'TA0002': 'from-blue-500 to-cyan-500',
            'TA0003': 'from-green-500 to-emerald-500',
            'TA0004': 'from-yellow-500 to-orange-500',
            'TA0005': 'from-red-500 to-rose-500',
            'TA0006': 'from-pink-500 to-fuchsia-500',
            'TA0007': 'from-cyan-500 to-blue-500',
            'TA0008': 'from-orange-500 to-amber-500',
            'TA0009': 'from-teal-500 to-cyan-500',
            'TA0010': 'from-indigo-500 to-purple-500',
            'TA0011': 'from-rose-500 to-red-500',
            'TA0040': 'from-red-600 to-orange-600',
        };
        return colors[tacticId] || 'from-slate-500 to-slate-600';
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                    MITRE ATT&CK Matrix
                </h1>
                <p className="text-slate-400 mt-1">
                    Adversary tactics and techniques mapped to detected threats
                </p>
            </div>

            {/* Tactics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {tactics.map((tactic) => {
                    const detectionCount = (tactic.techniques || []).reduce((sum, tid) => sum + (heatmap[tid] || 0), 0);

                    return (
                        <div
                            key={tactic.id}
                            onClick={() => loadTechniques(tactic.id)}
                            className={`
                                cursor-pointer p-6 rounded-xl border-2 transition-all duration-300 hover:scale-105
                                ${selectedTactic === tactic.id
                                    ? `bg-gradient-to-br ${getTacticColor(tactic.id)}/20 border-${getTacticColor(tactic.id).split('-')[1]}-500/50 shadow-2xl`
                                    : 'bg-slate-800/50 border-slate-700 hover:border-cyan-500/50'
                                }
                            `}
                        >
                            <div className="flex items-start justify-between mb-3">
                                <Shield className={`w-8 h-8 ${selectedTactic === tactic.id ? 'text-white' : 'text-cyan-400'}`} />
                                <span className="px-2 py-1 text-xs font-mono bg-slate-900/50 text-cyan-400 rounded border border-cyan-500/30">
                                    {tactic.id}
                                </span>
                            </div>
                            <h3 className="text-lg font-bold text-white mb-2">{tactic.name}</h3>
                            <p className="text-sm text-slate-400 mb-3 line-clamp-2">{tactic.description}</p>
                            <div className="flex items-center justify-between">
                                <span className="text-xs text-slate-500">
                                    {tactic.techniques?.length || 0} techniques
                                </span>
                                {detectionCount > 0 && (
                                    <span className="px-2 py-1 text-xs font-bold bg-red-500/20 text-red-400 rounded-full border border-red-500/50">
                                        {detectionCount} detections
                                    </span>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Techniques List */}
            {selectedTactic && techniques.length > 0 && (
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 shadow-2xl p-6">
                    <h2 className="text-2xl font-bold text-white mb-6">
                        Techniques for {tactics.find(t => t.id === selectedTactic)?.name}
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {techniques.map((technique) => {
                            const detectionCount = heatmap[technique.id] || 0;

                            return (
                                <div
                                    key={technique.id}
                                    className={`
                                        p-4 rounded-lg border transition-all
                                        ${detectionCount > 0
                                            ? `${getHeatmapIntensity(technique.id)} border-red-500/50 shadow-lg`
                                            : 'bg-slate-800/30 border-slate-700'
                                        }
                                    `}
                                >
                                    <div className="flex items-start justify-between mb-2">
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-2 mb-1">
                                                <span className="px-2 py-0.5 text-xs font-mono bg-slate-900/50 text-cyan-400 rounded">
                                                    {technique.id}
                                                </span>
                                                <h4 className="font-medium text-white">{technique.name}</h4>
                                            </div>
                                            <p className="text-sm text-slate-400">{technique.description}</p>
                                        </div>
                                        {detectionCount > 0 && (
                                            <span className="ml-4 px-3 py-1 text-sm font-bold bg-red-500/30 text-red-400 rounded-full border border-red-500/50 whitespace-nowrap">
                                                {detectionCount}×
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center space-x-2 mt-3">
                                        <span className={`px-2 py-0.5 text-xs font-medium rounded ${technique.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                                            technique.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                                                technique.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                                    'bg-blue-500/20 text-blue-400'
                                            }`}>
                                            {technique.severity}
                                        </span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Info Card */}
            {!selectedTactic && (
                <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 rounded-xl border border-cyan-500/30 p-6">
                    <div className="flex items-start space-x-4">
                        <Info className="w-6 h-6 text-cyan-400 flex-shrink-0 mt-1" />
                        <div>
                            <h3 className="text-lg font-bold text-white mb-2">About MITRE ATT&CK</h3>
                            <p className="text-slate-300 text-sm leading-relaxed">
                                The MITRE ATT&CK framework is a globally-accessible knowledge base of adversary tactics and techniques
                                based on real-world observations. Click on any tactic above to view techniques and see which ones
                                have been detected in your environment (highlighted in red).
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

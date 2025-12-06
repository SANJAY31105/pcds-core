'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/lib/api';
import { HuntQuery } from '@/types';
import { Search, Play, Clock, Target, AlertTriangle } from 'lucide-react';

export default function HuntPage() {
    const [queries, setQueries] = useState<HuntQuery[]>([]);
    const [loading, setLoading] = useState(true);
    const [runningQuery, setRunningQuery] = useState<string | null>(null);
    const [results, setResults] = useState<any>(null);

    useEffect(() => {
        loadQueries();
    }, []);

    const loadQueries = async () => {
        try {
            const data = await apiClient.getHuntQueries() as any;
            setQueries(data.queries || []);
        } catch (error) {
            console.error('Failed to load hunt queries:', error);
        } finally {
            setLoading(false);
        }
    };

    const runQuery = async (queryId: string) => {
        setRunningQuery(queryId);
        setResults(null);
        try {
            const data = await apiClient.runHuntQuery(queryId);
            setResults(data);
        } catch (error) {
            console.error('Failed to run query:', error);
        } finally {
            setRunningQuery(null);
        }
    };

    const getSeverityColor = (severity: string) => {
        const colors = {
            critical: 'from-red-500 to-rose-600',
            high: 'from-orange-500 to-amber-600',
            medium: 'from-yellow-500 to-orange-500',
            low: 'from-blue-500 to-cyan-500'
        };
        return colors[severity as keyof typeof colors] || colors.low;
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                    Threat Hunting
                </h1>
                <p className="text-slate-400 mt-1">
                    Proactive threat discovery with pre-built hunt packages
                </p>
            </div>

            {/* Hunt Queries Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {queries.map((query) => (
                    <div
                        key={query.id}
                        className="bg-gradient-to-br from-slate-800/80 to-slate-900/80 p-6 rounded-xl border border-cyan-500/20 shadow-xl hover:shadow-2xl hover:border-cyan-500/40 transition-all"
                    >
                        <div className="flex items-start justify-between mb-4">
                            <div className="flex-1">
                                <h3 className="text-lg font-bold text-white mb-2">{query.name}</h3>
                                <p className="text-slate-400 text-sm">{query.description}</p>
                            </div>
                            <span className={`px-3 py-1 text-xs font-bold rounded-full bg-gradient-to-r ${getSeverityColor(query.severity || 'medium')}/20 ${query.severity === 'critical' ? 'text-red-400' : query.severity === 'high' ? 'text-orange-400' : 'text-yellow-400'} border border-current`}>
                                {query.severity?.toUpperCase() || 'MEDIUM'}
                            </span>
                        </div>

                        {/* MITRE Techniques */}
                        <div className="mb-4">
                            <p className="text-xs text-slate-500 mb-2">MITRE ATT&CK Techniques:</p>
                            <div className="flex flex-wrap gap-2">
                                {(query.techniques || []).map((technique) => (
                                    <span
                                        key={technique}
                                        className="px-2 py-1 text-xs font-mono bg-slate-700/50 text-cyan-400 rounded border border-cyan-500/30"
                                    >
                                        {technique}
                                    </span>
                                ))}
                            </div>
                        </div>

                        {/* Run Button */}
                        <button
                            onClick={() => runQuery(query.id)}
                            disabled={runningQuery === query.id}
                            className="w-full px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-medium rounded-lg hover:shadow-lg hover:shadow-cyan-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                        >
                            {runningQuery === query.id ? (
                                <>
                                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                    <span>Running...</span>
                                </>
                            ) : (
                                <>
                                    <Play className="w-4 h-4" />
                                    <span>Run Hunt</span>
                                </>
                            )}
                        </button>
                    </div>
                ))}
            </div>

            {/* Results */}
            {results && (
                <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 rounded-xl border border-cyan-500/20 shadow-2xl p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-2xl font-bold text-white">Hunt Results</h2>
                        <div className="flex items-center space-x-4">
                            <div className="px-4 py-2 bg-green-500/20 rounded-lg border border-green-500/50">
                                <span className="text-green-400 font-medium">{results.total_findings} Findings</span>
                            </div>
                            <div className="text-sm text-slate-400">
                                {new Date(results.timestamp).toLocaleString()}
                            </div>
                        </div>
                    </div>

                    {!results.results || results.results.length === 0 ? (
                        <div className="text-center py-12">
                            <Target className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                            <p className="text-slate-400">No threats found matching this hunt query</p>
                            <p className="text-slate-500 text-sm mt-2">This is good news - your environment is clean!</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {results.results.map((threat: any, index: number) => (
                                <div
                                    key={threat.id || index}
                                    className="p-4 bg-slate-800/50 rounded-lg border border-slate-700 hover:border-cyan-500/30 transition-colors"
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex-1">
                                            <h4 className="font-medium text-white">{threat.title}</h4>
                                            <p className="text-sm text-slate-400 mt-1">{threat.description}</p>
                                            <div className="flex items-center space-x-4 mt-2 text-xs text-slate-500">
                                                <span>Source: {threat.source_ip}</span>
                                                <span>•</span>
                                                <span>Score: {threat.risk_score}</span>
                                                <span>•</span>
                                                <span>{new Date(threat.timestamp).toLocaleString()}</span>
                                            </div>
                                        </div>
                                        <span className={`px-3 py-1 text-xs font-bold rounded-full ${threat.severity === 'critical' ? 'bg-red-500/20 text-red-400 border border-red-500/50' :
                                            threat.severity === 'high' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/50' :
                                                threat.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50' :
                                                    'bg-blue-500/20 text-blue-400 border border-blue-500/50'
                                            }`}>
                                            {threat.severity?.toUpperCase()}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

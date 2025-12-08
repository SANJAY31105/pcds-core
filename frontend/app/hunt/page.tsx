'use client';

import { useState } from 'react';
import { Play, Target, Shield, Clock, ChevronRight, CheckCircle, Crosshair, Lock, Server, Globe, Radar } from 'lucide-react';

interface HuntQuery {
    id: string;
    name: string;
    description: string;
    severity: 'critical' | 'high' | 'medium';
    techniques: string[];
}

export default function HuntPage() {
    const [runningQuery, setRunningQuery] = useState<string | null>(null);
    const [completedQueries, setCompletedQueries] = useState<string[]>([]);
    const [results, setResults] = useState<any>(null);

    const huntQueries: HuntQuery[] = [
        { id: 'ransomware', name: 'Ransomware Precursors', description: 'Detect early ransomware indicators', severity: 'critical', techniques: ['T1486', 'T1490'] },
        { id: 'lateral', name: 'Lateral Movement', description: 'Identify credential abuse and pivoting', severity: 'high', techniques: ['T1021', 'T1550'] },
        { id: 'c2', name: 'C2 Beacon Detection', description: 'Find command and control patterns', severity: 'critical', techniques: ['T1071', 'T1573'] },
        { id: 'creds', name: 'Credential Theft', description: 'Hunt for credential dumping', severity: 'high', techniques: ['T1003', 'T1558'] },
        { id: 'exfil', name: 'Data Exfiltration', description: 'Detect large data transfers', severity: 'critical', techniques: ['T1041', 'T1567'] },
        { id: 'persistence', name: 'Persistence Mechanisms', description: 'Find registry and startup changes', severity: 'medium', techniques: ['T1547', 'T1053'] }
    ];

    const runQuery = async (queryId: string) => {
        if (runningQuery) return;
        setRunningQuery(queryId);
        setResults(null);

        await new Promise(resolve => setTimeout(resolve, 2000));

        const mockResults: Record<string, any[]> = {
            ransomware: [{ title: 'Suspicious vssadmin execution', entity: 'workstation-15', risk: 85 }],
            lateral: [{ title: 'PsExec from non-admin', entity: '192.168.1.50', risk: 72 }],
            c2: [{ title: 'Periodic beacon detected', entity: '192.168.1.100', risk: 88 }],
            creds: [],
            exfil: [{ title: 'Large cloud upload', entity: 'user-12', risk: 65 }],
            persistence: []
        };

        setResults({ queryId, findings: mockResults[queryId] || [], total: mockResults[queryId]?.length || 0 });
        setCompletedQueries(prev => [...prev, queryId]);
        setRunningQuery(null);
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white">Threat Hunting</h1>
                <p className="text-[#666] text-sm mt-1">Proactive threat discovery using MITRE ATT&CK techniques</p>
            </div>

            {/* Hunt Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {huntQueries.map((query) => {
                    const isRunning = runningQuery === query.id;
                    const isCompleted = completedQueries.includes(query.id);

                    return (
                        <div key={query.id} className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <div className="flex items-start justify-between mb-3">
                                <span className={`text-xs font-medium px-2 py-1 rounded ${query.severity === 'critical' ? 'bg-[#ef4444]/15 text-[#ef4444]' :
                                    query.severity === 'high' ? 'bg-[#f97316]/15 text-[#f97316]' :
                                        'bg-[#eab308]/15 text-[#eab308]'
                                    }`}>
                                    {query.severity.toUpperCase()}
                                </span>
                                {isCompleted && <CheckCircle className="w-4 h-4 text-[#22c55e]" />}
                            </div>

                            <h3 className="text-base font-medium text-white mb-1">{query.name}</h3>
                            <p className="text-sm text-[#666] mb-4">{query.description}</p>

                            <div className="flex gap-1.5 mb-4">
                                {query.techniques.map((tech) => (
                                    <span key={tech} className="px-2 py-0.5 text-xs font-mono bg-[#1a1a1a] text-[#10a37f] rounded">
                                        {tech}
                                    </span>
                                ))}
                            </div>

                            <button
                                onClick={() => runQuery(query.id)}
                                disabled={isRunning}
                                className={`w-full py-2.5 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors ${isRunning
                                    ? 'bg-[#1a1a1a] text-[#666] cursor-not-allowed'
                                    : 'bg-[#10a37f] text-white hover:bg-[#0d8a6a]'
                                    }`}
                            >
                                {isRunning ? (
                                    <>
                                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                                        Running...
                                    </>
                                ) : (
                                    <>
                                        <Play className="w-4 h-4" />
                                        Run Hunt
                                    </>
                                )}
                            </button>
                        </div>
                    );
                })}
            </div>

            {/* Results */}
            {results && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-base font-medium text-white">Hunt Results</h3>
                        <span className={`text-sm ${results.total > 0 ? 'text-[#ef4444]' : 'text-[#22c55e]'}`}>
                            {results.total > 0 ? `${results.total} findings` : 'No threats found'}
                        </span>
                    </div>

                    {results.findings.length > 0 ? (
                        <div className="space-y-2">
                            {results.findings.map((f: any, i: number) => (
                                <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a]">
                                    <div>
                                        <p className="text-sm text-white">{f.title}</p>
                                        <p className="text-xs text-[#666]">{f.entity}</p>
                                    </div>
                                    <span className="text-sm font-medium text-white">{f.risk}</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-8">
                            <Shield className="w-10 h-10 mx-auto mb-2 text-[#22c55e]/50" />
                            <p className="text-sm text-[#666]">Environment clear</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

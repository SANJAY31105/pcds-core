'use client';

import { useState, useEffect } from 'react';

interface SavedQuery {
    name: string;
    description: string;
    query: string;
    tags: string[];
    run_count: number;
}

interface HuntResult {
    timestamp: string;
    event_type: string;
    source: string;
    [key: string]: any;
}

interface IOC {
    indicator: string;
    type: string;
    threat: string;
    confidence: number;
}

export default function ThreatHuntingPage() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<HuntResult[]>([]);
    const [savedQueries, setSavedQueries] = useState<SavedQuery[]>([]);
    const [iocs, setIocs] = useState<IOC[]>([]);
    const [stats, setStats] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [iocLookup, setIocLookup] = useState('');
    const [iocResult, setIocResult] = useState<any>(null);
    const [activeTab, setActiveTab] = useState<'hunt' | 'ioc' | 'saved'>('hunt');

    const API_BASE = 'http://localhost:8000/api/v2';

    const fetchData = async () => {
        try {
            // Fetch saved queries
            const savedRes = await fetch(`${API_BASE}/hunt/saved-queries`);
            if (savedRes.ok) {
                const data = await savedRes.json();
                setSavedQueries(data.queries || []);
            }

            // Fetch IOCs
            const iocRes = await fetch(`${API_BASE}/hunt/ioc/list`);
            if (iocRes.ok) {
                const data = await iocRes.json();
                setIocs(data.iocs || []);
            }

            // Fetch stats
            const statsRes = await fetch(`${API_BASE}/hunt/stats`);
            if (statsRes.ok) {
                const data = await statsRes.json();
                setStats(data);
            }
        } catch (error) {
            console.error('Error fetching hunt data:', error);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const executeQuery = async () => {
        if (!query.trim()) return;

        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/hunt/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, time_range_hours: 24, limit: 100 })
            });

            if (res.ok) {
                const data = await res.json();
                setResults(data.results || []);
            }
        } catch (error) {
            console.error('Error executing query:', error);
        } finally {
            setLoading(false);
        }
    };

    const runSavedQuery = async (savedQuery: SavedQuery) => {
        setQuery(savedQuery.query);
        setActiveTab('hunt');

        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/hunt/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: savedQuery.query, time_range_hours: 24, limit: 100 })
            });

            if (res.ok) {
                const data = await res.json();
                setResults(data.results || []);
            }
        } catch (error) {
            console.error('Error running saved query:', error);
        } finally {
            setLoading(false);
        }
    };

    const lookupIOC = async () => {
        if (!iocLookup.trim()) return;

        try {
            const res = await fetch(`${API_BASE}/hunt/ioc/lookup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ indicator: iocLookup, indicator_type: 'auto' })
            });

            if (res.ok) {
                const data = await res.json();
                setIocResult(data);
            }
        } catch (error) {
            console.error('Error looking up IOC:', error);
        }
    };

    const simulateEvents = async () => {
        try {
            await fetch(`${API_BASE}/hunt/simulate?count=100`, { method: 'POST' });
            await fetchData();
            alert('✅ Generated 100 test events!');
        } catch (error) {
            console.error('Error simulating events:', error);
            alert('⚠️ Generated test events (demo mode)');
        }
    };

    const getTagColor = (tag: string) => {
        const colors: { [key: string]: string } = {
            'execution': 'bg-orange-500',
            'persistence': 'bg-red-500',
            'credential': 'bg-purple-500',
            'lateral': 'bg-yellow-500',
            'c2': 'bg-pink-500',
            'exfiltration': 'bg-red-600',
            'network': 'bg-blue-500',
            'lolbas': 'bg-green-500',
            'ransomware': 'bg-red-700',
            'injection': 'bg-orange-600',
        };
        return colors[tag] || 'bg-gray-500';
    };

    return (
        <div className="min-h-screen bg-[#0a0a0a] p-6 text-[#e5e5e5]">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-3xl font-semibold text-white mb-2">
                    🔍 Threat Hunting
                </h1>
                <p className="text-[#666]">
                    CrowdStrike-style query-based threat hunting across telemetry
                </p>
            </div>

            {/* Stats Bar */}
            <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Total Events</div>
                    <div className="text-2xl font-bold text-[#10a37f]">{stats?.total_events || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Saved Queries</div>
                    <div className="text-2xl font-bold text-[#eab308]">{stats?.saved_queries || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">IOC Database</div>
                    <div className="text-2xl font-bold text-[#f97316]">{stats?.iocs || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Event Types</div>
                    <div className="text-2xl font-bold text-[#3b82f6]">
                        {Object.keys(stats?.event_types || {}).length}
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setActiveTab('hunt')}
                    className={`px-4 py-2 rounded-lg font-medium transition ${activeTab === 'hunt'
                        ? 'bg-[#10a37f] text-white'
                        : 'bg-[#141414] text-[#a1a1a1] border border-[#2a2a2a] hover:bg-[#1a1a1a]'
                        }`}
                >
                    🔍 Hunt Query
                </button>
                <button
                    onClick={() => setActiveTab('saved')}
                    className={`px-4 py-2 rounded-lg font-medium transition ${activeTab === 'saved'
                        ? 'bg-[#10a37f] text-white'
                        : 'bg-[#141414] text-[#a1a1a1] border border-[#2a2a2a] hover:bg-[#1a1a1a]'
                        }`}
                >
                    📁 Saved Queries
                </button>
                <button
                    onClick={() => setActiveTab('ioc')}
                    className={`px-4 py-2 rounded-lg font-medium transition ${activeTab === 'ioc'
                        ? 'bg-[#10a37f] text-white'
                        : 'bg-[#141414] text-[#a1a1a1] border border-[#2a2a2a] hover:bg-[#1a1a1a]'
                        }`}
                >
                    🎯 IOC Lookup
                </button>
                <button
                    onClick={simulateEvents}
                    className="ml-auto px-4 py-2 bg-[#3b82f6] hover:bg-blue-600 rounded-lg text-white transition"
                >
                    Generate Test Events
                </button>
            </div>

            {/* Hunt Query Tab */}
            {activeTab === 'hunt' && (
                <div className="space-y-6">
                    {/* Query Input */}
                    <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                        <h3 className="text-white font-semibold mb-4">Hunt Query</h3>
                        <div className="flex gap-4">
                            <input
                                type="text"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && executeQuery()}
                                placeholder="event_type:process AND process_name:powershell"
                                className="flex-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-4 py-3 text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f]"
                            />
                            <button
                                onClick={executeQuery}
                                disabled={loading}
                                className="px-6 py-3 bg-[#10a37f] hover:bg-[#0d8a6a] rounded-lg text-white font-medium transition disabled:opacity-50"
                            >
                                {loading ? 'Searching...' : 'Hunt'}
                            </button>
                        </div>
                        <div className="mt-3 text-[#666] text-sm">
                            Syntax: <code className="text-[#10a37f]">field:value</code> |
                            <code className="text-[#10a37f] ml-2">field:*wildcard*</code> |
                            <code className="text-[#10a37f] ml-2">field:&gt;N</code> |
                            <code className="text-[#10a37f] ml-2">AND / NOT</code>
                        </div>
                    </div>

                    {/* Results */}
                    <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                        <h3 className="text-white font-semibold mb-4">
                            Results ({results.length})
                        </h3>
                        <div className="space-y-2 max-h-96 overflow-y-auto">
                            {results.length === 0 ? (
                                <div className="text-[#666] text-center py-8">
                                    No results. Enter a query or click "Generate Test Events" first.
                                </div>
                            ) : (
                                results.map((result, idx) => (
                                    <div
                                        key={idx}
                                        className="p-4 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a]"
                                    >
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <span className="px-2 py-1 bg-blue-600/20 text-blue-400 text-xs rounded border border-blue-600/30">
                                                    {result.event_type}
                                                </span>
                                                <span className="text-[#a1a1a1] text-xs ml-2">{result.source}</span>
                                            </div>
                                            <span className="text-[#666] text-xs">
                                                {new Date(result.timestamp).toLocaleString()}
                                            </span>
                                        </div>
                                        <div className="mt-2 text-[#e5e5e5] text-sm font-mono">
                                            {JSON.stringify(
                                                Object.fromEntries(
                                                    Object.entries(result).filter(
                                                        ([k]) => !['timestamp', 'event_type', 'source'].includes(k)
                                                    )
                                                ),
                                                null,
                                                2
                                            )}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Saved Queries Tab */}
            {activeTab === 'saved' && (
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h3 className="text-white font-semibold mb-4">Pre-built Hunt Queries</h3>
                    <div className="grid gap-4">
                        {savedQueries.map((sq, idx) => (
                            <div
                                key={idx}
                                className="p-4 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a] hover:border-[#333] cursor-pointer transition"
                                onClick={() => runSavedQuery(sq)}
                            >
                                <div className="flex justify-between items-start">
                                    <div>
                                        <h4 className="text-white font-medium">{sq.name}</h4>
                                        <p className="text-[#666] text-sm mt-1">{sq.description}</p>
                                        <code className="text-[#10a37f] text-xs mt-2 block">{sq.query}</code>
                                    </div>
                                    <button className="px-3 py-1 bg-[#10a37f] hover:bg-[#0d8a6a] rounded text-white text-sm">
                                        Run
                                    </button>
                                </div>
                                <div className="flex gap-2 mt-3">
                                    {sq.tags.map((tag, tidx) => (
                                        <span
                                            key={tidx}
                                            className={`px-2 py-0.5 text-xs text-white rounded ${getTagColor(tag)}`}
                                        >
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* IOC Lookup Tab */}
            {activeTab === 'ioc' && (
                <div className="space-y-6">
                    {/* IOC Lookup */}
                    <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                        <h3 className="text-white font-semibold mb-4">IOC Lookup</h3>
                        <div className="flex gap-4">
                            <input
                                type="text"
                                value={iocLookup}
                                onChange={(e) => setIocLookup(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && lookupIOC()}
                                placeholder="Enter IP, domain, or hash..."
                                className="flex-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-4 py-3 text-white placeholder-[#666] focus:outline-none focus:border-[#10a37f]"
                            />
                            <button
                                onClick={lookupIOC}
                                className="px-6 py-3 bg-[#f97316] hover:bg-orange-600 rounded-lg text-white font-medium transition"
                            >
                                Lookup
                            </button>
                        </div>

                        {/* IOC Result */}
                        {iocResult && (
                            <div className={`mt-4 p-4 rounded-lg border ${iocResult.found ? 'bg-red-900/20 border-red-500/50' : 'bg-[#1a1a1a] border-[#333]'}`}>
                                {iocResult.found ? (
                                    <div>
                                        <div className="flex items-center gap-2 mb-2">
                                            <span className="text-red-400 text-xl">⚠️</span>
                                            <span className="text-white font-bold">MALICIOUS IOC DETECTED</span>
                                        </div>
                                        <div className="space-y-1 text-sm">
                                            <div><span className="text-[#666]">Indicator:</span> <span className="text-white">{iocResult.indicator}</span></div>
                                            <div><span className="text-[#666]">Type:</span> <span className="text-[#10a37f]">{iocResult.type}</span></div>
                                            <div><span className="text-[#666]">Threat:</span> <span className="text-red-400">{iocResult.threat}</span></div>
                                            <div><span className="text-[#666]">Confidence:</span> <span className="text-white">{(iocResult.confidence * 100).toFixed(0)}%</span></div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="text-[#10a37f]">
                                        ✓ Indicator not found in threat database
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* IOC Database */}
                    <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                        <h3 className="text-white font-semibold mb-4">IOC Database ({iocs.length})</h3>
                        <div className="space-y-2 max-h-64 overflow-y-auto">
                            {iocs.map((ioc, idx) => (
                                <div
                                    key={idx}
                                    className="flex justify-between items-center p-3 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a]"
                                >
                                    <div>
                                        <span className="text-white font-mono">{ioc.indicator}</span>
                                        <span className="text-[#666] text-xs ml-2">({ioc.type})</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="text-[#ef4444] text-sm">{ioc.threat}</span>
                                        <span className="text-[#666] text-xs">{(ioc.confidence * 100).toFixed(0)}%</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Footer */}
            <div className="mt-6 text-center text-[#666] text-sm">
                PCDS Threat Hunting • 8 Pre-built Queries • IOC Database • Real-time Telemetry
            </div>
        </div>
    );
}

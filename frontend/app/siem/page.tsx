'use client';

import { useState, useEffect } from 'react';

interface SIEMConfig {
    name: string;
    enabled: boolean;
    transport: string;
    format: string;
    host: string;
    port: number;
    min_severity: string;
    stats?: {
        sent: number;
        failed: number;
        last_sent: string | null;
    };
}

interface FormatInfo {
    id: string;
    name: string;
    description: string;
}

interface TransportInfo {
    id: string;
    name: string;
    port?: number;
    description?: string;
}

export default function SIEMPage() {
    const [configs, setConfigs] = useState<SIEMConfig[]>([]);
    const [formats, setFormats] = useState<FormatInfo[]>([]);
    const [transports, setTransports] = useState<TransportInfo[]>([]);
    const [stats, setStats] = useState<any>(null);
    const [showAddForm, setShowAddForm] = useState(false);
    const [testResult, setTestResult] = useState<any>(null);

    // New config form
    const [newConfig, setNewConfig] = useState({
        name: '',
        transport: 'syslog_udp',
        format: 'syslog',
        host: '',
        port: 514,
        min_severity: 'medium',
        token: '',
        webhook_url: ''
    });

    const API_BASE = 'http://localhost:8000/api/v2';

    const fetchData = async () => {
        try {
            // Fetch configs
            const configRes = await fetch(`${API_BASE}/siem/config`);
            if (configRes.ok) {
                const data = await configRes.json();
                setConfigs(data.configs || []);
            }

            // Fetch formats
            const formatsRes = await fetch(`${API_BASE}/siem/formats`);
            if (formatsRes.ok) {
                const data = await formatsRes.json();
                setFormats(data.formats || []);
                setTransports(data.transports || []);
            }

            // Fetch stats
            const statsRes = await fetch(`${API_BASE}/siem/stats`);
            if (statsRes.ok) {
                const data = await statsRes.json();
                setStats(data);
            }
        } catch (error) {
            console.error('Error fetching SIEM data:', error);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const addConfig = async () => {
        try {
            const res = await fetch(`${API_BASE}/siem/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ...newConfig,
                    enabled: true
                })
            });

            if (res.ok) {
                setShowAddForm(false);
                setNewConfig({
                    name: '',
                    transport: 'syslog_udp',
                    format: 'syslog',
                    host: '',
                    port: 514,
                    min_severity: 'medium',
                    token: '',
                    webhook_url: ''
                });
                fetchData();
            }
        } catch (error) {
            console.error('Error adding config:', error);
        }
    };

    const deleteConfig = async (name: string) => {
        try {
            await fetch(`${API_BASE}/siem/config/${name}`, { method: 'DELETE' });
            fetchData();
        } catch (error) {
            console.error('Error deleting config:', error);
        }
    };

    const testConnection = async (name: string) => {
        try {
            const res = await fetch(`${API_BASE}/siem/test/${name}`, { method: 'POST' });
            if (res.ok) {
                const data = await res.json();
                setTestResult({ name, ...data });
                setTimeout(() => setTestResult(null), 5000);
            }
        } catch (error) {
            setTestResult({ name, success: false, error: 'Connection failed' });
        }
    };

    const getTransportIcon = (transport: string) => {
        switch (transport) {
            case 'syslog_udp': return '📡';
            case 'syslog_tcp': return '🔌';
            case 'webhook': return '🌐';
            case 'splunk_hec': return '⚡';
            default: return '📤';
        }
    };

    const getFormatBadge = (format: string) => {
        const colors: { [key: string]: string } = {
            'syslog': 'bg-[#3b82f6]',
            'cef': 'bg-[#a855f7]',
            'leef': 'bg-[#f97316]',
            'json': 'bg-[#22c55e]'
        };
        return colors[format] || 'bg-[#666]';
    };

    return (
        <div className="min-h-screen bg-[#0a0a0a] p-6 space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white">SIEM Integration</h1>
                <p className="text-[#666] text-sm mt-1">Export alerts to Splunk, Elastic, QRadar and other SIEM platforms</p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Total Sent</div>
                    <div className="text-2xl font-bold text-[#22c55e]">{stats?.total_sent || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Failed</div>
                    <div className="text-2xl font-bold text-[#ef4444]">{stats?.total_failed || 0}</div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Success Rate</div>
                    <div className="text-2xl font-bold text-[#3b82f6]">
                        {((stats?.success_rate || 0) * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                    <div className="text-[#666] text-sm">Destinations</div>
                    <div className="text-2xl font-bold text-[#a855f7]">
                        {stats?.enabled || 0}/{stats?.destinations || 0}
                    </div>
                </div>
            </div>

            {/* Test Result Toast */}
            {testResult && (
                <div className={`mb-4 p-4 rounded-lg border ${testResult.success ? 'bg-green-900/20 border-green-500/50' : 'bg-red-900/20 border-red-500/50'}`}>
                    <span className={testResult.success ? 'text-[#22c55e]' : 'text-[#ef4444]'}>
                        {testResult.success ? '✓' : '✗'} Test to "{testResult.name}": {testResult.success ? 'Success' : 'Failed'}
                    </span>
                </div>
            )}

            {/* Add Button */}
            <button
                onClick={() => setShowAddForm(!showAddForm)}
                className="mb-4 px-4 py-2 bg-[#10a37f] hover:bg-[#0d8a6a] rounded-lg text-white font-medium transition"
            >
                + Add SIEM Destination
            </button>

            {/* Add Form */}
            {showAddForm && (
                <div className="mb-6 bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h3 className="text-white font-semibold mb-4">New SIEM Configuration</h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="text-[#666] text-sm">Name</label>
                            <input
                                type="text"
                                value={newConfig.name}
                                onChange={(e) => setNewConfig({ ...newConfig, name: e.target.value })}
                                className="w-full mt-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white placeholder-[#666]"
                                placeholder="my_splunk"
                            />
                        </div>
                        <div>
                            <label className="text-[#666] text-sm">Transport</label>
                            <select
                                value={newConfig.transport}
                                onChange={(e) => setNewConfig({ ...newConfig, transport: e.target.value })}
                                className="w-full mt-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white"
                            >
                                {transports.map((t) => (
                                    <option key={t.id} value={t.id}>{t.name}</option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <label className="text-[#666] text-sm">Format</label>
                            <select
                                value={newConfig.format}
                                onChange={(e) => setNewConfig({ ...newConfig, format: e.target.value })}
                                className="w-full mt-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white"
                            >
                                {formats.map((f) => (
                                    <option key={f.id} value={f.id}>{f.name}</option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <label className="text-[#666] text-sm">Min Severity</label>
                            <select
                                value={newConfig.min_severity}
                                onChange={(e) => setNewConfig({ ...newConfig, min_severity: e.target.value })}
                                className="w-full mt-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white"
                            >
                                <option value="low">Low</option>
                                <option value="medium">Medium</option>
                                <option value="high">High</option>
                                <option value="critical">Critical</option>
                            </select>
                        </div>
                        <div>
                            <label className="text-[#666] text-sm">Host</label>
                            <input
                                type="text"
                                value={newConfig.host}
                                onChange={(e) => setNewConfig({ ...newConfig, host: e.target.value })}
                                className="w-full mt-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white placeholder-[#666]"
                                placeholder="splunk.local"
                            />
                        </div>
                        <div>
                            <label className="text-[#666] text-sm">Port</label>
                            <input
                                type="number"
                                value={newConfig.port}
                                onChange={(e) => setNewConfig({ ...newConfig, port: parseInt(e.target.value) })}
                                className="w-full mt-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white"
                            />
                        </div>
                        {newConfig.transport === 'splunk_hec' && (
                            <div className="col-span-2">
                                <label className="text-[#666] text-sm">HEC Token</label>
                                <input
                                    type="password"
                                    value={newConfig.token}
                                    onChange={(e) => setNewConfig({ ...newConfig, token: e.target.value })}
                                    className="w-full mt-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white placeholder-[#666]"
                                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                                />
                            </div>
                        )}
                        {newConfig.transport === 'webhook' && (
                            <div className="col-span-2">
                                <label className="text-[#666] text-sm">Webhook URL</label>
                                <input
                                    type="text"
                                    value={newConfig.webhook_url}
                                    onChange={(e) => setNewConfig({ ...newConfig, webhook_url: e.target.value })}
                                    className="w-full mt-1 bg-[#1a1a1a] border border-[#333] rounded-lg px-3 py-2 text-white placeholder-[#666]"
                                    placeholder="https://hooks.example.com/..."
                                />
                            </div>
                        )}
                    </div>
                    <div className="mt-4 flex gap-2">
                        <button
                            onClick={addConfig}
                            className="px-4 py-2 bg-[#22c55e] hover:bg-green-700 rounded-lg text-white transition"
                        >
                            Save
                        </button>
                        <button
                            onClick={() => setShowAddForm(false)}
                            className="px-4 py-2 bg-[#1a1a1a] border border-[#333] hover:bg-[#222] rounded-lg text-white transition"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}

            {/* Configurations List */}
            <div className="space-y-4">
                {configs.length === 0 ? (
                    <div className="bg-[#141414] rounded-xl p-8 text-center text-[#666] border border-[#2a2a2a]">
                        No SIEM destinations configured. Click "Add SIEM Destination" to get started.
                    </div>
                ) : (
                    configs.map((config) => (
                        <div
                            key={config.name}
                            className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]"
                        >
                            <div className="flex justify-between items-start">
                                <div className="flex items-center gap-3">
                                    <span className="text-2xl">{getTransportIcon(config.transport)}</span>
                                    <div>
                                        <h3 className="text-white font-semibold">{config.name}</h3>
                                        <p className="text-[#666] text-sm">
                                            {config.host}:{config.port}
                                        </p>
                                    </div>
                                    <span className={`px-2 py-1 text-xs text-white rounded ${getFormatBadge(config.format)}`}>
                                        {config.format.toUpperCase()}
                                    </span>
                                    <span className={`px-2 py-1 text-xs rounded ${config.enabled ? 'bg-[#22c55e] text-white' : 'bg-[#666] text-[#333]'}`}>
                                        {config.enabled ? 'Enabled' : 'Disabled'}
                                    </span>
                                </div>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => testConnection(config.name)}
                                        className="px-3 py-1 bg-[#3b82f6] hover:bg-blue-700 rounded text-white text-sm transition"
                                    >
                                        Test
                                    </button>
                                    <button
                                        onClick={() => deleteConfig(config.name)}
                                        className="px-3 py-1 bg-[#ef4444] hover:bg-red-700 rounded text-white text-sm transition"
                                    >
                                        Delete
                                    </button>
                                </div>
                            </div>

                            {/* Stats */}
                            {config.stats && (
                                <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                                    <div>
                                        <span className="text-[#666]">Sent:</span>
                                        <span className="text-[#22c55e] ml-2">{config.stats.sent}</span>
                                    </div>
                                    <div>
                                        <span className="text-[#666]">Failed:</span>
                                        <span className="text-[#ef4444] ml-2">{config.stats.failed}</span>
                                    </div>
                                    <div>
                                        <span className="text-[#666]">Last:</span>
                                        <span className="text-[#a1a1a1] ml-2">
                                            {config.stats.last_sent ? new Date(config.stats.last_sent).toLocaleString() : 'Never'}
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>

            {/* Supported Formats */}
            <div className="mt-8 bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                <h3 className="text-white font-semibold mb-4">Supported Formats & Transports</h3>
                <div className="grid grid-cols-2 gap-6">
                    <div>
                        <h4 className="text-[#666] text-sm mb-2">Formats</h4>
                        <div className="space-y-2">
                            {formats.map((f) => (
                                <div key={f.id} className="flex items-center gap-2">
                                    <div className={`w-2 h-2 rounded-full ${getFormatBadge(f.id)}`} />
                                    <span className="text-white">{f.name}</span>
                                    <span className="text-[#666] text-sm">- {f.description}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div>
                        <h4 className="text-[#666] text-sm mb-2">Transports</h4>
                        <div className="space-y-2">
                            {transports.map((t) => (
                                <div key={t.id} className="flex items-center gap-2">
                                    <span className="text-lg">{getTransportIcon(t.id)}</span>
                                    <span className="text-white">{t.name}</span>
                                    {t.port && <span className="text-[#666] text-sm">Port {t.port}</span>}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="mt-6 text-center text-[#666] text-sm">
                PCDS SIEM Integration • Syslog RFC 5424 • CEF • LEEF • Splunk HEC
            </div>
        </div>
    );
}

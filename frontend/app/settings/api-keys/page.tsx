'use client';

import { useState, useEffect } from 'react';
import { Key, Copy, Plus, Trash2, Check, Shield, AlertCircle } from 'lucide-react';

interface ApiKey {
    id: string;
    key_masked: string;
    name: string;
    created_at: string;
    last_used: string | null;
    is_active: boolean;
}

export default function ApiKeysPage() {
    const [keys, setKeys] = useState<ApiKey[]>([]);
    const [newKeyName, setNewKeyName] = useState('');
    const [generatedKey, setGeneratedKey] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Get user email from localStorage (set during login)
    const userEmail = typeof window !== 'undefined'
        ? localStorage.getItem('userEmail') || 'demo@pcdsai.app'
        : 'demo@pcdsai.app';

    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://pcds-backend-production.up.railway.app';

    const fetchKeys = async () => {
        try {
            const res = await fetch(`${API_URL}/api/v2/keys?user_email=${encodeURIComponent(userEmail)}`);
            const data = await res.json();
            setKeys(data.keys || []);
        } catch (err) {
            console.error('Failed to fetch keys:', err);
        }
    };

    useEffect(() => {
        fetchKeys();
    }, []);

    const generateKey = async () => {
        setLoading(true);
        setError(null);

        try {
            const res = await fetch(`${API_URL}/api/v2/keys/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_email: userEmail,
                    key_name: newKeyName || 'Default Agent Key'
                })
            });

            const data = await res.json();

            if (data.success) {
                setGeneratedKey(data.key);
                setNewKeyName('');
                fetchKeys();
            } else {
                setError('Failed to generate key');
            }
        } catch (err) {
            setError('Connection error');
        }

        setLoading(false);
    };

    const revokeKey = async (keyId: string) => {
        if (!confirm('Are you sure you want to revoke this key? The agent will stop working.')) return;

        try {
            await fetch(`${API_URL}/api/v2/keys/${keyId}`, { method: 'DELETE' });
            fetchKeys();
        } catch (err) {
            console.error('Failed to revoke key:', err);
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="min-h-screen bg-[#0a0a0a] text-white p-8">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="flex items-center gap-3 mb-8">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 flex items-center justify-center border border-emerald-500/30">
                        <Key className="w-6 h-6 text-emerald-400" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold">API Keys</h1>
                        <p className="text-gray-400 text-sm">Manage keys for your PCDS agents</p>
                    </div>
                </div>

                {/* Generated Key Alert */}
                {generatedKey && (
                    <div className="mb-6 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
                        <div className="flex items-start gap-3">
                            <Shield className="w-5 h-5 text-emerald-400 mt-0.5" />
                            <div className="flex-1">
                                <h3 className="font-semibold text-emerald-400 mb-1">Key Generated Successfully!</h3>
                                <p className="text-sm text-gray-400 mb-3">
                                    Copy this key now. It won't be shown again.
                                </p>
                                <div className="flex items-center gap-2 bg-black/40 rounded-lg p-3 font-mono text-sm">
                                    <code className="flex-1 text-emerald-300 break-all">{generatedKey}</code>
                                    <button
                                        onClick={() => copyToClipboard(generatedKey)}
                                        className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                                    >
                                        {copied ? (
                                            <Check className="w-4 h-4 text-emerald-400" />
                                        ) : (
                                            <Copy className="w-4 h-4 text-gray-400" />
                                        )}
                                    </button>
                                </div>
                            </div>
                            <button
                                onClick={() => setGeneratedKey(null)}
                                className="text-gray-500 hover:text-white"
                            >
                                ×
                            </button>
                        </div>
                    </div>
                )}

                {/* Generate New Key */}
                <div className="mb-8 p-6 bg-[#111111] border border-gray-800 rounded-xl">
                    <h2 className="text-lg font-semibold mb-4">Generate New Key</h2>
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={newKeyName}
                            onChange={(e) => setNewKeyName(e.target.value)}
                            placeholder="Key name (e.g., 'Production Server')"
                            className="flex-1 px-4 py-3 bg-black/40 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-emerald-500/50"
                        />
                        <button
                            onClick={generateKey}
                            disabled={loading}
                            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50"
                        >
                            <Plus className="w-4 h-4" />
                            {loading ? 'Generating...' : 'Generate Key'}
                        </button>
                    </div>
                    {error && (
                        <p className="mt-2 text-red-400 text-sm flex items-center gap-1">
                            <AlertCircle className="w-4 h-4" /> {error}
                        </p>
                    )}
                </div>

                {/* Existing Keys */}
                <div className="bg-[#111111] border border-gray-800 rounded-xl overflow-hidden">
                    <div className="p-4 border-b border-gray-800">
                        <h2 className="text-lg font-semibold">Your API Keys</h2>
                    </div>

                    {keys.length === 0 ? (
                        <div className="p-8 text-center text-gray-500">
                            <Key className="w-12 h-12 mx-auto mb-3 opacity-30" />
                            <p>No API keys yet. Generate one to get started.</p>
                        </div>
                    ) : (
                        <div className="divide-y divide-gray-800">
                            {keys.map((key) => (
                                <div key={key.id} className="p-4 flex items-center justify-between hover:bg-white/5">
                                    <div className="flex items-center gap-4">
                                        <div className={`w-2 h-2 rounded-full ${key.is_active ? 'bg-emerald-400' : 'bg-red-400'}`} />
                                        <div>
                                            <p className="font-medium">{key.name}</p>
                                            <p className="text-sm text-gray-500 font-mono">{key.key_masked}</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div className="text-right text-sm text-gray-500">
                                            <p>Created: {new Date(key.created_at).toLocaleDateString()}</p>
                                            {key.last_used && (
                                                <p>Last used: {new Date(key.last_used).toLocaleDateString()}</p>
                                            )}
                                        </div>
                                        {key.is_active && (
                                            <button
                                                onClick={() => revokeKey(key.id)}
                                                className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                                                title="Revoke key"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Help Section */}
                <div className="mt-8 p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                    <h3 className="font-medium text-blue-400 mb-2">How to use your API key</h3>
                    <ol className="text-sm text-gray-400 space-y-1 list-decimal list-inside">
                        <li>Download the PCDS Agent from the <a href="/download" className="text-blue-400 hover:underline">Download page</a></li>
                        <li>Run the installer and paste your API key when prompted</li>
                        <li>The agent will start monitoring and sending data to your dashboard</li>
                    </ol>
                </div>
            </div>
        </div>
    );
}

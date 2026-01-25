'use client';

import { useState, useEffect } from 'react';
import { Shield, Download, Copy, Check, Terminal, Server, Activity } from 'lucide-react';
import { useAuth } from '@/lib/AuthContext';

export default function AgentPage() {
    const { user } = useAuth();
    const [apiKey, setApiKey] = useState('');
    const [customerId, setCustomerId] = useState('');
    const [copied, setCopied] = useState(false);
    const [generating, setGenerating] = useState(false);
    const [events, setEvents] = useState<any[]>([]);

    const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://pcds-backend-production.up.railway.app';

    const generateApiKey = async () => {
        setGenerating(true);
        try {
            const res = await fetch(`${API_BASE}/api/v2/customers/api-key`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ customer_name: user?.email || 'Customer' })
            });
            const data = await res.json();
            setApiKey(data.api_key);
            setCustomerId(data.customer_id);
        } catch (err) {
            console.error(err);
        } finally {
            setGenerating(false);
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    // Fetch events for demo customer
    useEffect(() => {
        const fetchEvents = async () => {
            try {
                const res = await fetch(`${API_BASE}/api/v2/customers/demo/events?limit=10`);
                const data = await res.json();
                setEvents(data.events || []);
            } catch (err) {
                console.error(err);
            }
        };
        fetchEvents();
        const interval = setInterval(fetchEvents, 5000);
        return () => clearInterval(interval);
    }, []);

    const installCommand = `pip install psutil requests && python pcds_agent.py --api-key ${apiKey || 'YOUR_API_KEY'}`;

    return (
        <div className="space-y-8 max-w-4xl">
            <div>
                <h1 className="text-2xl font-bold text-white mb-2">PCDS Agent</h1>
                <p className="text-[#888]">Install the agent on your network to start detecting threats</p>
            </div>

            {/* Step 1: Generate API Key */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-8 h-8 rounded-full bg-[#10a37f]/20 flex items-center justify-center text-[#10a37f] font-bold">1</div>
                    <h2 className="text-lg font-semibold text-white">Generate Your API Key</h2>
                </div>

                {!apiKey ? (
                    <button
                        onClick={generateApiKey}
                        disabled={generating}
                        className="px-6 py-3 bg-[#10a37f] hover:bg-[#0d8a6a] text-white rounded-lg transition-colors disabled:opacity-50"
                    >
                        {generating ? 'Generating...' : 'Generate API Key'}
                    </button>
                ) : (
                    <div className="space-y-3">
                        <div className="flex items-center gap-2">
                            <code className="flex-1 bg-[#0a0a0a] px-4 py-3 rounded-lg text-[#10a37f] font-mono text-sm">
                                {apiKey}
                            </code>
                            <button
                                onClick={() => copyToClipboard(apiKey)}
                                className="p-3 bg-[#0a0a0a] rounded-lg hover:bg-[#1a1a1a] transition-colors"
                            >
                                {copied ? <Check className="w-5 h-5 text-[#10a37f]" /> : <Copy className="w-5 h-5 text-[#666]" />}
                            </button>
                        </div>
                        <p className="text-xs text-yellow-500">⚠️ Save this key! It won't be shown again.</p>
                    </div>
                )}
            </div>

            {/* Step 2: Download Agent */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-8 h-8 rounded-full bg-[#10a37f]/20 flex items-center justify-center text-[#10a37f] font-bold">2</div>
                    <h2 className="text-lg font-semibold text-white">Download the Agent</h2>
                </div>

                <a
                    href="https://raw.githubusercontent.com/your-repo/pcds-core/main/agent/pcds_agent.py"
                    download="pcds_agent.py"
                    className="inline-flex items-center gap-2 px-6 py-3 bg-[#1a1a1a] hover:bg-[#222] text-white rounded-lg transition-colors border border-[#2a2a2a]"
                >
                    <Download className="w-5 h-5" />
                    Download pcds_agent.py
                </a>
                <p className="text-xs text-[#666] mt-3">Works on Windows, Linux, and macOS</p>
            </div>

            {/* Step 3: Run Agent */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-8 h-8 rounded-full bg-[#10a37f]/20 flex items-center justify-center text-[#10a37f] font-bold">3</div>
                    <h2 className="text-lg font-semibold text-white">Run the Agent</h2>
                </div>

                <div className="bg-[#0a0a0a] rounded-lg p-4 font-mono text-sm">
                    <div className="flex items-center gap-2 text-[#666] mb-2">
                        <Terminal className="w-4 h-4" />
                        <span>Terminal</span>
                    </div>
                    <code className="text-[#10a37f] break-all">{installCommand}</code>
                </div>
                <button
                    onClick={() => copyToClipboard(installCommand)}
                    className="mt-3 text-sm text-[#10a37f] hover:underline"
                >
                    Copy command
                </button>
            </div>

            {/* Live Events */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <Activity className="w-5 h-5 text-[#10a37f]" />
                        <h2 className="text-lg font-semibold text-white">Incoming Events</h2>
                    </div>
                    <span className="text-xs text-[#666]">{events.length} events (demo)</span>
                </div>

                {events.length > 0 ? (
                    <div className="space-y-2 max-h-[300px] overflow-y-auto">
                        {events.slice(-10).reverse().map((event, i) => (
                            <div key={i} className="flex items-center justify-between p-3 bg-[#0a0a0a] rounded-lg text-sm">
                                <div className="flex items-center gap-3">
                                    <Server className="w-4 h-4 text-[#666]" />
                                    <span className="text-white">{event.source_ip}</span>
                                    <span className="text-[#666]">→</span>
                                    <span className="text-[#888]">{event.dest_ip || 'N/A'}:{event.dest_port || ''}</span>
                                </div>
                                <span className="text-xs text-[#666]">{event.process_name || 'unknown'}</span>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="text-center py-8 text-[#666]">
                        <Server className="w-10 h-10 mx-auto mb-3 opacity-50" />
                        <p>No events yet. Run the agent to see real-time data.</p>
                    </div>
                )}
            </div>
        </div>
    );
}

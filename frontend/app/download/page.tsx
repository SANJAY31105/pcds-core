// PCDS Download Page - Last updated: 2026-01-30
'use client';

import { useState } from 'react';
import { Download, Shield, Clock, CheckCircle, Smartphone, Copy, Check, Key } from 'lucide-react';

export default function DownloadPage() {
    const [trialStarted, setTrialStarted] = useState(false);
    const [email, setEmail] = useState('');
    const [apiKey, setApiKey] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [copied, setCopied] = useState(false);

    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://pcds-backend-production.up.railway.app';

    const handleStartTrial = async () => {
        if (!email || !email.includes('@')) {
            setError('Please enter a valid email address');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const res = await fetch(`${API_URL}/api/v2/keys/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_email: email,
                    key_name: 'Trial Agent Key'
                })
            });

            const data = await res.json();

            if (data.success && data.key) {
                setApiKey(data.key);
                setTrialStarted(true);
                // Save email for later
                localStorage.setItem('userEmail', email);
            } else {
                // Fallback to demo key if API not ready
                setApiKey('pcds_demo_key_12345');
                setTrialStarted(true);
                setError('Using demo key (API deploying)');
            }
        } catch (err) {
            // Fallback to demo key on error
            console.error('API Error:', err);
            setApiKey('pcds_demo_key_12345');
            setTrialStarted(true);
            setError('Using demo key (API deploying)');
        }

        setLoading(false);
    };

    const copyToClipboard = () => {
        if (apiKey) {
            navigator.clipboard.writeText(apiKey);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    return (
        <div className="max-w-5xl mx-auto py-12 px-6">
            <h1 className="text-4xl font-bold text-white mb-4">Deploy PCDS Agent</h1>
            <p className="text-[#a1a1a1] text-lg mb-12">
                Secure your infrastructure in minutes. Start your 14-day free trial today.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Windows Agent Card */}
                <div className="bg-[#141414] border border-[#2a2a2a] rounded-2xl p-8 hover:border-[#22c55e]/50 transition-colors relative overflow-hidden">
                    <div className="absolute top-0 right-0 bg-[#22c55e]/10 px-4 py-1 rounded-bl-xl border-l border-b border-[#22c55e]/20">
                        <span className="text-[#22c55e] text-xs font-bold uppercase tracking-wider">Recommended</span>
                    </div>

                    <div className="w-16 h-16 bg-[#22c55e]/10 rounded-xl flex items-center justify-center mb-6">
                        <Shield className="w-8 h-8 text-[#22c55e]" />
                    </div>

                    <h2 className="text-2xl font-bold text-white mb-2">Windows Enterprise Agent</h2>
                    <p className="text-[#666] mb-6">
                        Full-spectrum protection for Windows Servers and Workstations (10/11/Server 2019+).
                        Includes Anti-Ransomware, kill-switch, and real-time monitoring.
                    </p>

                    <div className="space-y-3 mb-8">
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#22c55e]" />
                            <span className="text-sm text-[#a1a1a1]">Active Defense (Kill Switch)</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#22c55e]" />
                            <span className="text-sm text-[#a1a1a1]">Zero-Impact Performance</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#22c55e]" />
                            <span className="text-sm text-[#a1a1a1]">Instant Dashboard Connection</span>
                        </div>
                    </div>

                    {!trialStarted ? (
                        <div className="space-y-4">
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="Enter your work email"
                                className="w-full px-4 py-3 bg-black/40 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-[#22c55e]/50"
                            />
                            <button
                                onClick={handleStartTrial}
                                disabled={loading}
                                className="w-full bg-[#22c55e] hover:bg-[#16a34a] text-white font-bold py-4 rounded-xl flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                            >
                                <Clock className="w-5 h-5" />
                                {loading ? 'Generating...' : 'Start 14-Day Free Trial'}
                            </button>
                            {error && !trialStarted && (
                                <p className="text-red-400 text-sm text-center">{error}</p>
                            )}
                        </div>
                    ) : (
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4">
                            {/* API Key Display */}
                            <div className="p-4 bg-[#22c55e]/10 border border-[#22c55e]/20 rounded-xl">
                                <div className="flex items-center gap-2 mb-2">
                                    <Key className="w-4 h-4 text-[#22c55e]" />
                                    <p className="text-[#22c55e] text-sm font-medium">Your API Key</p>
                                </div>
                                <div className="flex items-center gap-2 bg-black/40 rounded-lg p-3">
                                    <code className="flex-1 text-white text-sm font-mono break-all">{apiKey}</code>
                                    <button
                                        onClick={copyToClipboard}
                                        className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                                    >
                                        {copied ? (
                                            <Check className="w-4 h-4 text-[#22c55e]" />
                                        ) : (
                                            <Copy className="w-4 h-4 text-gray-400" />
                                        )}
                                    </button>
                                </div>
                                <p className="text-[#666] text-xs mt-2">Copy this key. You'll need it during installation.</p>
                            </div>

                            {error && (
                                <p className="text-yellow-400 text-xs text-center">{error}</p>
                            )}

                            <a
                                href="/PCDS_Setup.exe"
                                download="PCDS_Setup.exe"
                                className="w-full bg-white hover:bg-gray-100 text-black font-bold py-4 rounded-xl flex items-center justify-center gap-2 transition-all"
                            >
                                <Download className="w-5 h-5" />
                                Download Installer (64-bit)
                            </a>

                            <p className="text-[#666] text-xs text-center">
                                Run the installer and paste your API key when prompted
                            </p>
                        </div>
                    )}
                </div>

                {/* Mobile App Card */}
                <div className="bg-[#141414] border border-[#2a2a2a] rounded-2xl p-8 opacity-75 hover:opacity-100 transition-opacity">
                    <div className="w-16 h-16 bg-[#3b82f6]/10 rounded-xl flex items-center justify-center mb-6">
                        <Smartphone className="w-8 h-8 text-[#3b82f6]" />
                    </div>

                    <h2 className="text-2xl font-bold text-white mb-2">Mobile Companion</h2>
                    <p className="text-[#666] mb-6">
                        Monitor threats on the go. Receive push notifications for critical alerts.
                    </p>

                    <div className="p-6 border border-[#2a2a2a] border-dashed rounded-xl flex items-center justify-center text-center">
                        <p className="text-[#666] text-sm">
                            Scan QR code on Dashboard<br />to install PWA
                        </p>
                    </div>
                </div>
            </div>

            {/* Quick Setup Guide */}
            <div className="mt-12 p-6 bg-[#141414] border border-[#2a2a2a] rounded-2xl">
                <h3 className="text-xl font-bold text-white mb-4">Quick Setup Guide</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="flex gap-4">
                        <div className="w-8 h-8 rounded-full bg-[#22c55e]/20 flex items-center justify-center text-[#22c55e] font-bold shrink-0">1</div>
                        <div>
                            <h4 className="text-white font-medium">Enter Email</h4>
                            <p className="text-[#666] text-sm">Get your unique API key instantly</p>
                        </div>
                    </div>
                    <div className="flex gap-4">
                        <div className="w-8 h-8 rounded-full bg-[#22c55e]/20 flex items-center justify-center text-[#22c55e] font-bold shrink-0">2</div>
                        <div>
                            <h4 className="text-white font-medium">Download & Install</h4>
                            <p className="text-[#666] text-sm">Run the installer, paste your key</p>
                        </div>
                    </div>
                    <div className="flex gap-4">
                        <div className="w-8 h-8 rounded-full bg-[#22c55e]/20 flex items-center justify-center text-[#22c55e] font-bold shrink-0">3</div>
                        <div>
                            <h4 className="text-white font-medium">View Dashboard</h4>
                            <p className="text-[#666] text-sm">Data appears in seconds</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

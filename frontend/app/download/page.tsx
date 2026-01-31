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

                {/* macOS Agent Card */}
                <div className="bg-[#141414] border border-[#2a2a2a] rounded-2xl p-8 hover:border-[#a855f7]/50 transition-colors">
                    <div className="w-16 h-16 bg-[#a855f7]/10 rounded-xl flex items-center justify-center mb-6">
                        <svg className="w-8 h-8 text-[#a855f7]" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z" />
                        </svg>
                    </div>

                    <h2 className="text-2xl font-bold text-white mb-2">macOS Agent</h2>
                    <p className="text-[#666] mb-6">
                        Enterprise protection for Mac workstations (macOS 12+). Lightweight daemon with real-time monitoring.
                    </p>

                    <div className="space-y-3 mb-8">
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#a855f7]" />
                            <span className="text-sm text-[#a1a1a1]">Native Apple Silicon Support</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#a855f7]" />
                            <span className="text-sm text-[#a1a1a1]">Minimal CPU Usage (~1%)</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#a855f7]" />
                            <span className="text-sm text-[#a1a1a1]">Instant Dashboard Connection</span>
                        </div>
                    </div>

                    {trialStarted ? (
                        <div className="space-y-3">
                            <div className="p-3 bg-[#a855f7]/10 border border-[#a855f7]/20 rounded-lg">
                                <p className="text-[#a855f7] text-xs font-mono mb-1">Quick Install (Terminal):</p>
                                <code className="text-white text-xs break-all">curl -sL https://pcdsai.app/install.sh | bash</code>
                            </div>
                            <a
                                href="/pcds-agent-mac.zip"
                                download
                                className="w-full bg-[#a855f7] hover:bg-[#9333ea] text-white font-bold py-3 rounded-xl flex items-center justify-center gap-2 transition-all"
                            >
                                <Download className="w-5 h-5" />
                                Download for Mac
                            </a>
                        </div>
                    ) : (
                        <p className="text-[#666] text-sm text-center py-4">Enter email above to get API key first</p>
                    )}
                </div>
            </div>

            {/* Second Row - Linux and Mobile */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
                {/* Linux Agent Card */}
                <div className="bg-[#141414] border border-[#2a2a2a] rounded-2xl p-8 hover:border-[#f97316]/50 transition-colors">
                    <div className="w-16 h-16 bg-[#f97316]/10 rounded-xl flex items-center justify-center mb-6">
                        <svg className="w-8 h-8 text-[#f97316]" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12.504 0c-.155 0-.315.008-.48.021-4.226.333-3.105 4.807-3.17 6.298-.076 1.092-.3 1.953-1.05 3.02-.885 1.051-2.127 2.75-2.716 4.521-.278.832-.41 1.684-.287 2.489a.424.424 0 00-.11.135c-.26.268-.45.6-.663.839-.199.199-.485.267-.797.4-.313.136-.658.269-.864.68-.09.189-.136.394-.132.602 0 .199.027.4.055.536.058.399.116.728.04.97-.249.68-.28 1.145-.106 1.484.174.334.535.47.94.601.81.2 1.91.135 2.774.6.926.466 1.866.67 2.616.47.526-.116.97-.464 1.208-.946.587.26 1.22.39 1.827.39.676 0 1.295-.145 1.833-.43.244.49.712.845 1.266.96.74.2 1.675-.005 2.592-.46.863-.465 1.962-.4 2.774-.6.405-.13.766-.27.94-.6.174-.34.142-.804-.107-1.485-.073-.242-.015-.57.04-.97.028-.135.055-.337.055-.536a1.35 1.35 0 00-.131-.601c-.21-.412-.553-.545-.865-.68-.312-.134-.598-.202-.798-.4-.213-.24-.403-.572-.664-.84a.424.424 0 00-.11-.134c.122-.805-.01-1.657-.287-2.49-.581-1.77-1.83-3.47-2.716-4.52-.751-1.068-.975-1.93-1.051-3.021-.064-1.491 1.057-5.965-3.169-6.298A8.702 8.702 0 0012.504 0z" />
                        </svg>
                    </div>

                    <h2 className="text-2xl font-bold text-white mb-2">Linux Agent</h2>
                    <p className="text-[#666] mb-6">
                        Server-grade protection for Linux (Ubuntu 20.04+, RHEL 8+, Debian 11+). Systemd service with logging.
                    </p>

                    <div className="space-y-3 mb-8">
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#f97316]" />
                            <span className="text-sm text-[#a1a1a1]">Systemd Service Integration</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#f97316]" />
                            <span className="text-sm text-[#a1a1a1]">Headless Server Support</span>
                        </div>
                        <div className="flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-[#f97316]" />
                            <span className="text-sm text-[#a1a1a1]">Container-Ready (Docker/K8s)</span>
                        </div>
                    </div>

                    {trialStarted ? (
                        <div className="space-y-3">
                            <div className="p-3 bg-[#f97316]/10 border border-[#f97316]/20 rounded-lg">
                                <p className="text-[#f97316] text-xs font-mono mb-1">Quick Install (Terminal):</p>
                                <code className="text-white text-xs break-all">curl -sL https://pcdsai.app/install.sh | sudo bash</code>
                            </div>
                            <a
                                href="/pcds-agent-linux.tar.gz"
                                download
                                className="w-full bg-[#f97316] hover:bg-[#ea580c] text-white font-bold py-3 rounded-xl flex items-center justify-center gap-2 transition-all"
                            >
                                <Download className="w-5 h-5" />
                                Download for Linux
                            </a>
                        </div>
                    ) : (
                        <p className="text-[#666] text-sm text-center py-4">Enter email above to get API key first</p>
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

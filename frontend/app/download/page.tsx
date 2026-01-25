'use client';

import { useState } from 'react';
import { Download, Shield, Clock, CheckCircle, Smartphone } from 'lucide-react';

export default function DownloadPage() {
    const [trialStarted, setTrialStarted] = useState(false);

    const handleStartTrial = () => {
        // In real app, this would call API to create trial license
        setTrialStarted(true);
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
                        <button
                            onClick={handleStartTrial}
                            className="w-full bg-[#22c55e] hover:bg-[#16a34a] text-white font-bold py-4 rounded-xl flex items-center justify-center gap-2 transition-all"
                        >
                            <Clock className="w-5 h-5" />
                            Start 14-Day Free Trial
                        </button>
                    ) : (
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4">
                            <div className="p-4 bg-[#22c55e]/10 border border-[#22c55e]/20 rounded-xl">
                                <p className="text-[#22c55e] text-sm font-medium mb-1">Trial Active</p>
                                <p className="text-white text-xs">Your license key has been generated automatically.</p>
                            </div>
                            <a
                                href="/PCDS_Setup.exe"
                                download="PCDS_Setup.exe"
                                className="w-full bg-white hover:bg-gray-100 text-black font-bold py-4 rounded-xl flex items-center justify-center gap-2 transition-all"
                            >
                                <Download className="w-5 h-5" />
                                Download Installer (64-bit)
                            </a>
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
        </div>
    );
}

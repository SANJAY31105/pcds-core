'use client';

import { useState } from 'react';

interface URLAnalysis {
    is_phishing: boolean;
    confidence: number;
    risk_score: number;
    indicators: string[];
    recommendation: string;
}

interface EmailAnalysis {
    is_phishing: boolean;
    confidence: number;
    risk_score: number;
    indicators: string[];
    recommendation: string;
}

export default function PhishingPage() {
    const [activeTab, setActiveTab] = useState<'url' | 'email'>('url');
    const [url, setUrl] = useState('');
    const [email, setEmail] = useState({ content: '', sender: '', subject: '' });
    const [urlResult, setUrlResult] = useState<URLAnalysis | null>(null);
    const [emailResult, setEmailResult] = useState<EmailAnalysis | null>(null);
    const [loading, setLoading] = useState(false);
    const [stats, setStats] = useState<any>(null);

    const API_BASE = 'http://localhost:8000/api/v2/advanced-ml';

    const checkURL = async () => {
        if (!url) return;
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/phishing/check-url`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            const data = await res.json();
            setUrlResult(data);
            fetchStats();
        } catch (error) {
            console.error('Failed to check URL:', error);
        }
        setLoading(false);
    };

    const checkEmail = async () => {
        if (!email.content) return;
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/phishing/check-email`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(email)
            });
            const data = await res.json();
            setEmailResult(data);
            fetchStats();
        } catch (error) {
            console.error('Failed to check email:', error);
        }
        setLoading(false);
    };

    const fetchStats = async () => {
        try {
            const res = await fetch(`${API_BASE}/phishing/stats`);
            const data = await res.json();
            setStats(data);
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    };

    const getRiskColor = (score: number) => {
        if (score >= 70) return 'text-[#ef4444]';
        if (score >= 40) return 'text-[#eab308]';
        return 'text-[#22c55e]';
    };

    const getRiskBg = (score: number) => {
        if (score >= 70) return 'bg-[#ef4444]';
        if (score >= 40) return 'bg-[#eab308]';
        return 'bg-[#22c55e]';
    };

    const examplePhishingURL = 'http://paypa1-secure.tk/login?account=verify&urgent=1';
    const examplePhishingEmail = `Dear Customer,

Your account has been suspended due to unusual activity! You must verify your identity immediately.

Click here to verify: http://secure-bank.xyz/login

Failure to verify within 24 hours will result in permanent account suspension.

Security Team`;

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white flex items-center gap-3">
                    🎣 Phishing Scanner
                </h1>
                <p className="text-[#666] text-sm mt-1">
                    AI-powered detection of phishing URLs and emails using NLP analysis
                </p>
            </div>

            {/* Stats */}
            {stats && (
                <div className="grid grid-cols-3 gap-4">
                    <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                        <div className="text-[#666] text-sm">URLs Analyzed</div>
                        <div className="text-2xl font-bold text-white">{stats.urls_analyzed || 0}</div>
                    </div>
                    <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                        <div className="text-[#666] text-sm">Emails Analyzed</div>
                        <div className="text-2xl font-bold text-white">{stats.emails_analyzed || 0}</div>
                    </div>
                    <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                        <div className="text-[#666] text-sm">Phishing Detected</div>
                        <div className="text-2xl font-bold text-[#ef4444]">{stats.phishing_detected || 0}</div>
                    </div>
                </div>
            )}

            {/* Tabs */}
            <div className="flex gap-2">
                <button
                    onClick={() => setActiveTab('url')}
                    className={`px-6 py-3 rounded-lg font-medium transition ${activeTab === 'url'
                        ? 'bg-[#10a37f] text-white'
                        : 'bg-[#141414] text-[#888] hover:bg-[#1a1a1a] border border-[#2a2a2a]'
                        }`}
                >
                    🔗 Check URL
                </button>
                <button
                    onClick={() => setActiveTab('email')}
                    className={`px-6 py-3 rounded-lg font-medium transition ${activeTab === 'email'
                        ? 'bg-[#10a37f] text-white'
                        : 'bg-[#141414] text-[#888] hover:bg-[#1a1a1a] border border-[#2a2a2a]'
                        }`}
                >
                    ✉️ Check Email
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Input Panel */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    {activeTab === 'url' ? (
                        <>
                            <h2 className="text-lg font-medium text-white mb-4">🔗 URL Analysis</h2>
                            <input
                                type="text"
                                placeholder="Enter URL to check..."
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg px-4 py-3 text-white mb-4 focus:border-[#10a37f] outline-none"
                            />
                            <div className="flex gap-3">
                                <button
                                    onClick={() => setUrl(examplePhishingURL)}
                                    className="flex-1 bg-[#1a1a1a] hover:bg-[#252525] text-white py-2 px-4 rounded-lg transition text-sm border border-[#2a2a2a]"
                                >
                                    📝 Load Example
                                </button>
                                <button
                                    onClick={checkURL}
                                    disabled={loading || !url}
                                    className="flex-1 bg-[#10a37f] hover:bg-[#0d8c6d] text-white py-2 px-4 rounded-lg transition disabled:opacity-50"
                                >
                                    {loading ? '⏳ Scanning...' : '🔍 Scan URL'}
                                </button>
                            </div>
                        </>
                    ) : (
                        <>
                            <h2 className="text-lg font-medium text-white mb-4">✉️ Email Analysis</h2>
                            <div className="space-y-3 mb-4">
                                <input
                                    type="text"
                                    placeholder="Sender email..."
                                    value={email.sender}
                                    onChange={(e) => setEmail({ ...email, sender: e.target.value })}
                                    className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg px-4 py-2 text-white text-sm focus:border-[#10a37f] outline-none"
                                />
                                <input
                                    type="text"
                                    placeholder="Subject..."
                                    value={email.subject}
                                    onChange={(e) => setEmail({ ...email, subject: e.target.value })}
                                    className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg px-4 py-2 text-white text-sm focus:border-[#10a37f] outline-none"
                                />
                                <textarea
                                    placeholder="Email content..."
                                    value={email.content}
                                    onChange={(e) => setEmail({ ...email, content: e.target.value })}
                                    rows={8}
                                    className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg px-4 py-3 text-white text-sm resize-none focus:border-[#10a37f] outline-none"
                                />
                            </div>
                            <div className="flex gap-3">
                                <button
                                    onClick={() => setEmail({
                                        sender: 'security@bankofamerica.scam.com',
                                        subject: 'URGENT: Account Suspended',
                                        content: examplePhishingEmail
                                    })}
                                    className="flex-1 bg-[#1a1a1a] hover:bg-[#252525] text-white py-2 px-4 rounded-lg transition text-sm border border-[#2a2a2a]"
                                >
                                    📝 Load Example
                                </button>
                                <button
                                    onClick={checkEmail}
                                    disabled={loading || !email.content}
                                    className="flex-1 bg-[#10a37f] hover:bg-[#0d8c6d] text-white py-2 px-4 rounded-lg transition disabled:opacity-50"
                                >
                                    {loading ? '⏳ Scanning...' : '🔍 Scan Email'}
                                </button>
                            </div>
                        </>
                    )}
                </div>

                {/* Results Panel */}
                <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                    <h2 className="text-lg font-medium text-white mb-4">📊 Analysis Results</h2>
                    {(activeTab === 'url' ? urlResult : emailResult) ? (
                        <div className="space-y-4">
                            <div className={`rounded-lg p-6 text-center ${(activeTab === 'url' ? urlResult?.is_phishing : emailResult?.is_phishing)
                                ? 'bg-[#ef4444]/20 border border-[#ef4444]'
                                : 'bg-[#22c55e]/20 border border-[#22c55e]'
                                }`}>
                                <div className="text-5xl mb-2">
                                    {(activeTab === 'url' ? urlResult?.is_phishing : emailResult?.is_phishing) ? '⚠️' : '✅'}
                                </div>
                                <div className={`text-2xl font-bold ${(activeTab === 'url' ? urlResult?.is_phishing : emailResult?.is_phishing)
                                    ? 'text-[#ef4444]'
                                    : 'text-[#22c55e]'
                                    }`}>
                                    {(activeTab === 'url' ? urlResult?.is_phishing : emailResult?.is_phishing) ? 'PHISHING DETECTED' : 'APPEARS SAFE'}
                                </div>
                            </div>

                            <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                <div className="flex justify-between items-center mb-2">
                                    <span className="text-[#666]">Risk Score</span>
                                    <span className={`text-2xl font-bold ${getRiskColor((activeTab === 'url' ? urlResult?.risk_score : emailResult?.risk_score) || 0)}`}>
                                        {(activeTab === 'url' ? urlResult?.risk_score : emailResult?.risk_score) || 0}/100
                                    </span>
                                </div>
                                <div className="w-full bg-[#1a1a1a] rounded-full h-3">
                                    <div
                                        className={`h-3 rounded-full transition-all ${getRiskBg((activeTab === 'url' ? urlResult?.risk_score : emailResult?.risk_score) || 0)}`}
                                        style={{ width: `${(activeTab === 'url' ? urlResult?.risk_score : emailResult?.risk_score) || 0}%` }}
                                    />
                                </div>
                            </div>

                            {((activeTab === 'url' ? urlResult?.indicators : emailResult?.indicators) || []).length > 0 && (
                                <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                    <h3 className="text-[#10a37f] font-medium mb-3">🚩 Suspicious Indicators</h3>
                                    <ul className="space-y-2">
                                        {((activeTab === 'url' ? urlResult?.indicators : emailResult?.indicators) || []).map((ind, i) => (
                                            <li key={i} className="flex items-start gap-2 text-white text-sm">
                                                <span className="text-[#ef4444]">•</span>
                                                {ind}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                                <h3 className="text-[#10a37f] font-medium mb-2">💡 Recommendation</h3>
                                <p className="text-white text-sm">
                                    {(activeTab === 'url' ? urlResult?.recommendation : emailResult?.recommendation)}
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="text-center text-[#666] py-12">
                            <div className="text-6xl mb-4">🔍</div>
                            <p>Enter a URL or email to analyze</p>
                            <p className="text-sm mt-2">Our AI will detect phishing attempts</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Tips */}
            <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                <h3 className="text-lg font-medium text-white mb-4">🛡️ Phishing Prevention Tips</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                        <div className="text-2xl mb-2">🔗</div>
                        <div className="text-white font-medium">Check URLs Carefully</div>
                        <div className="text-[#666] text-sm mt-1">
                            Hover over links before clicking. Look for misspellings and suspicious domains.
                        </div>
                    </div>
                    <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                        <div className="text-2xl mb-2">⚡</div>
                        <div className="text-white font-medium">Beware of Urgency</div>
                        <div className="text-[#666] text-sm mt-1">
                            Phishing often creates false urgency. Take your time to verify requests.
                        </div>
                    </div>
                    <div className="bg-[#0a0a0a] rounded-lg p-4 border border-[#2a2a2a]">
                        <div className="text-2xl mb-2">📞</div>
                        <div className="text-white font-medium">Verify Independently</div>
                        <div className="text-[#666] text-sm mt-1">
                            Contact organizations directly using official channels, not links in emails.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

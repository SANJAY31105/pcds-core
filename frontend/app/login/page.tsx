'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Shield, Lock, Mail, AlertCircle } from 'lucide-react';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const router = useRouter();

    const API_BASE = 'http://localhost:8000/api/v2';

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const res = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ email, password })
            });

            const data = await res.json();

            if (!res.ok) {
                setError(data.detail || 'Login failed');
                setLoading(false);
                return;
            }

            localStorage.setItem('access_token', data.access_token);
            localStorage.setItem('user', JSON.stringify(data.user));
            router.push('/');
        } catch (err) {
            setError('Connection error. Is the backend running?');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-[#10a37f]/10 border border-[#10a37f]/30 rounded-xl mb-4">
                        <Shield className="w-8 h-8 text-[#10a37f]" />
                    </div>
                    <h1 className="text-2xl font-semibold text-white">PCDS Enterprise</h1>
                    <p className="text-[#888] text-sm mt-1">Predictive Cyber Defense System</p>
                </div>

                {/* Login Form */}
                <div className="bg-[#141414] rounded-lg border border-[#2a2a2a] p-6">
                    <form onSubmit={handleLogin} className="space-y-5">
                        {error && (
                            <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">
                                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                {error}
                            </div>
                        )}

                        <div>
                            <label className="block text-sm text-[#888] mb-2">
                                Email
                            </label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="admin@pcds.com"
                                    className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg pl-10 pr-4 py-2.5 text-white placeholder-[#555] focus:outline-none focus:border-[#10a37f] transition-colors"
                                    required
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm text-[#888] mb-2">
                                Password
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg pl-10 pr-4 py-2.5 text-white placeholder-[#555] focus:outline-none focus:border-[#10a37f] transition-colors"
                                    required
                                />
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-[#10a37f] hover:bg-[#0d8a6a] text-white font-medium py-2.5 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                    </svg>
                                    Signing in...
                                </span>
                            ) : 'Sign In'}
                        </button>
                    </form>

                    {/* Demo Credentials */}
                    <div className="mt-6 pt-5 border-t border-[#2a2a2a]">
                        <p className="text-[#666] text-xs text-center mb-3">Quick Access</p>
                        <div className="grid grid-cols-2 gap-2">
                            <button
                                onClick={() => { setEmail('admin@pcds.com'); setPassword('admin123'); }}
                                className="bg-[#1a1a1a] hover:bg-[#222] text-[#888] hover:text-white text-xs py-2 px-3 rounded-lg transition-colors border border-[#2a2a2a]"
                            >
                                👑 Demo Admin
                            </button>
                            <button
                                onClick={() => { setEmail('analyst@pcds.com'); setPassword('analyst123'); }}
                                className="bg-[#1a1a1a] hover:bg-[#222] text-[#888] hover:text-white text-xs py-2 px-3 rounded-lg transition-colors border border-[#2a2a2a]"
                            >
                                🔍 Demo Analyst
                            </button>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="text-center mt-6">
                    <p className="text-[#555] text-xs">
                        Powered by Microsoft Azure AI
                    </p>
                </div>
            </div>
        </div>
    );
}

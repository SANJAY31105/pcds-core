'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Shield, Lock, Mail, AlertCircle, UserPlus } from 'lucide-react';
import { useAuth } from '@/lib/AuthContext';

import NetworkBackground from '@/components/NetworkBackground';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [loading, setLoading] = useState(false);
    const [isSignup, setIsSignup] = useState(false);
    const router = useRouter();
    const { login, signup } = useAuth();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setLoading(true);

        try {
            if (isSignup) {
                const result = await signup(email, password);
                if (result.success) {
                    setSuccess('Account created! Check your email to verify, then sign in.');
                    setIsSignup(false);
                } else {
                    setError(result.error || 'Signup failed');
                }
            } else {
                const success = await login(email, password);
                if (success) {
                    router.push('/dashboard');
                } else {
                    setError('Invalid email or password');
                }
            }
        } catch (err) {
            setError('Connection error. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#020617] flex items-center justify-center p-4 relative overflow-hidden">
            {/* Animated Background */}
            <div className="absolute inset-0 z-0">
                <NetworkBackground />
            </div>

            <div className="w-full max-w-md relative z-10">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-white/5 border border-white/10 backdrop-blur-sm rounded-2xl mb-4 shadow-[0_0_40px_-10px_rgba(245,193,108,0.3)]">
                        <Shield className="w-8 h-8 text-[#f5c16c]" />
                    </div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">PCDS Enterprise</h1>
                    <p className="text-gray-400 text-sm mt-2">Predictive Cyber Defense System</p>
                </div>

                {/* Login/Signup Form */}
                <div className="bg-[#111827]/60 backdrop-blur-xl rounded-2xl border border-white/10 p-8 shadow-2xl">
                    {/* Tab Toggle */}
                    <div className="flex mb-8 bg-black/20 rounded-xl p-1 border border-white/5">
                        <button
                            onClick={() => { setIsSignup(false); setError(''); setSuccess(''); }}
                            className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-semibold transition-all duration-200 ${!isSignup ? 'bg-white/10 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}
                        >
                            Sign In
                        </button>
                        <button
                            onClick={() => { setIsSignup(true); setError(''); setSuccess(''); }}
                            className={`flex-1 py-2.5 px-4 rounded-lg text-sm font-semibold transition-all duration-200 ${isSignup ? 'bg-white/10 text-white shadow-sm' : 'text-gray-500 hover:text-gray-300'}`}
                        >
                            Sign Up
                        </button>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        {error && (
                            <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 text-red-200 px-4 py-3 rounded-lg text-sm">
                                <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-500" />
                                {error}
                            </div>
                        )}

                        {success && (
                            <div className="flex items-center gap-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-200 px-4 py-3 rounded-lg text-sm">
                                <UserPlus className="w-5 h-5 flex-shrink-0 text-emerald-500" />
                                {success}
                            </div>
                        )}

                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Email Address
                            </label>
                            <div className="relative group">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 group-focus-within:text-[#f5c16c] transition-colors" />
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="name@company.com"
                                    className="w-full bg-black/40 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-[#f5c16c]/50 focus:ring-1 focus:ring-[#f5c16c]/50 transition-all"
                                    required
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Password {isSignup && <span className="text-gray-500 font-normal">(min 6 chars)</span>}
                            </label>
                            <div className="relative group">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 group-focus-within:text-[#f5c16c] transition-colors" />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    minLength={isSignup ? 6 : undefined}
                                    className="w-full bg-black/40 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-gray-600 focus:outline-none focus:border-[#f5c16c]/50 focus:ring-1 focus:ring-[#f5c16c]/50 transition-all"
                                    required
                                />
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full font-bold py-3.5 px-4 rounded-xl transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                            style={{
                                background: 'linear-gradient(180deg, #fde68a, #f5c16c)',
                                color: '#020617',
                                boxShadow: '0 4px 20px rgba(245, 193, 108, 0.25)'
                            }}
                        >
                            {loading ? (
                                <span className="flex items-center justify-center gap-2">
                                    <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                    </svg>
                                    {isSignup ? 'Creating account...' : 'Authenticating...'}
                                </span>
                            ) : (
                                isSignup ? 'Create Account' : 'Sign In to Dashboard'
                            )}
                        </button>
                    </form>
                </div>

                {/* Footer */}
                <div className="text-center mt-8">
                    <p className="text-gray-600 text-xs">
                        Protected by <span className="text-gray-500 font-semibold">Supabase Auth</span> & <span className="text-gray-500 font-semibold">PCDS Shield</span>
                    </p>
                </div>
            </div>
        </div>
    );
}

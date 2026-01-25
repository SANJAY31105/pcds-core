'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Shield, Lock, Mail, AlertCircle, UserPlus } from 'lucide-react';
import { useAuth } from '@/lib/AuthContext';

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

                {/* Login/Signup Form */}
                <div className="bg-[#141414] rounded-lg border border-[#2a2a2a] p-6">
                    {/* Tab Toggle */}
                    <div className="flex mb-6 bg-[#0a0a0a] rounded-lg p-1">
                        <button
                            onClick={() => { setIsSignup(false); setError(''); setSuccess(''); }}
                            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${!isSignup ? 'bg-[#10a37f] text-white' : 'text-[#666] hover:text-white'}`}
                        >
                            Sign In
                        </button>
                        <button
                            onClick={() => { setIsSignup(true); setError(''); setSuccess(''); }}
                            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${isSignup ? 'bg-[#10a37f] text-white' : 'text-[#666] hover:text-white'}`}
                        >
                            Sign Up
                        </button>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-5">
                        {error && (
                            <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">
                                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                {error}
                            </div>
                        )}

                        {success && (
                            <div className="flex items-center gap-2 bg-green-500/10 border border-green-500/30 text-green-400 px-4 py-3 rounded-lg text-sm">
                                <UserPlus className="w-4 h-4 flex-shrink-0" />
                                {success}
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
                                    placeholder="you@example.com"
                                    className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg pl-10 pr-4 py-2.5 text-white placeholder-[#555] focus:outline-none focus:border-[#10a37f] transition-colors"
                                    required
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm text-[#888] mb-2">
                                Password {isSignup && <span className="text-[#555]">(min 6 characters)</span>}
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    minLength={isSignup ? 6 : undefined}
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
                                    {isSignup ? 'Creating account...' : 'Signing in...'}
                                </span>
                            ) : (
                                isSignup ? 'Create Account' : 'Sign In'
                            )}
                        </button>
                    </form>
                </div>

                {/* Footer */}
                <div className="text-center mt-6">
                    <p className="text-[#555] text-xs">
                        Secured by Supabase Authentication
                    </p>
                </div>
            </div>
        </div>
    );
}

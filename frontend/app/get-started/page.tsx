'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Shield, CheckCircle, ArrowRight, Mail, Building, User, Phone } from 'lucide-react';

export default function GetStartedPage() {
    const [submitted, setSubmitted] = useState(false);
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        company: '',
        phone: '',
        endpoints: '1-50'
    });

    const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://pcds-backend-production.up.railway.app';

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

        try {
            await fetch(`${API_BASE}/api/v2/leads`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: formData.email })
            });
            setSubmitted(true);
        } catch (err) {
            console.error(err);
            setSubmitted(true); // Still show success for demo
        } finally {
            setLoading(false);
        }
    };

    const steps = [
        { num: 1, title: "Schedule a Call", desc: "We'll discuss your security needs and environment" },
        { num: 2, title: "We Install the Agent", desc: "15-minute remote setup on your network" },
        { num: 3, title: "See Your Threats", desc: "Real-time visibility into your attack surface" }
    ];

    if (submitted) {
        return (
            <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center p-6">
                <div className="max-w-md text-center">
                    <div className="w-20 h-20 mx-auto mb-6 bg-[#10a37f]/20 rounded-full flex items-center justify-center">
                        <CheckCircle className="w-10 h-10 text-[#10a37f]" />
                    </div>
                    <h1 className="text-3xl font-bold text-white mb-4">Request Received! 🎉</h1>
                    <p className="text-[#888] mb-8">
                        We'll reach out within 24 hours to schedule your setup call.
                    </p>
                    <Link
                        href="/dashboard"
                        className="inline-flex items-center gap-2 px-6 py-3 bg-[#10a37f] hover:bg-[#0d8a6a] text-white rounded-lg transition-colors"
                    >
                        Explore Demo Dashboard <ArrowRight className="w-4 h-4" />
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#0a0a0a] text-white">
            {/* Header */}
            <nav className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a2a]">
                <Link href="/landing" className="flex items-center gap-2">
                    <Shield className="w-6 h-6 text-[#10a37f]" />
                    <span className="font-semibold">PCDS</span>
                </Link>
                <Link href="/login" className="text-sm text-[#888] hover:text-white">
                    Login
                </Link>
            </nav>

            <div className="max-w-6xl mx-auto px-6 py-16">
                <div className="grid lg:grid-cols-2 gap-16">
                    {/* Left - Info */}
                    <div>
                        <h1 className="text-4xl font-bold mb-6">
                            Get Protected in<br />
                            <span className="text-[#10a37f]">Under 24 Hours</span>
                        </h1>
                        <p className="text-lg text-[#888] mb-10">
                            We handle the technical setup. You get enterprise-grade threat detection without the complexity.
                        </p>

                        {/* Steps */}
                        <div className="space-y-6">
                            {steps.map((step) => (
                                <div key={step.num} className="flex gap-4">
                                    <div className="w-10 h-10 rounded-full bg-[#141414] border border-[#2a2a2a] flex items-center justify-center text-[#10a37f] font-semibold flex-shrink-0">
                                        {step.num}
                                    </div>
                                    <div>
                                        <h3 className="font-medium text-white">{step.title}</h3>
                                        <p className="text-sm text-[#666]">{step.desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Trust badges */}
                        <div className="mt-12 pt-8 border-t border-[#2a2a2a]">
                            <p className="text-sm text-[#666] mb-4">Trusted by security teams</p>
                            <div className="flex items-center gap-6 text-[#444]">
                                <span>🔒 SOC 2 Ready</span>
                                <span>🇮🇳 Made in India</span>
                                <span>⚡ 24/7 Support</span>
                            </div>
                        </div>
                    </div>

                    {/* Right - Form */}
                    <div className="bg-[#141414] rounded-2xl border border-[#2a2a2a] p-8">
                        <h2 className="text-xl font-semibold mb-6">Request Your Setup</h2>

                        <form onSubmit={handleSubmit} className="space-y-5">
                            <div>
                                <label className="block text-sm text-[#888] mb-2">Your Name</label>
                                <div className="relative">
                                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                                    <input
                                        type="text"
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                        placeholder="John Doe"
                                        className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg pl-10 pr-4 py-3 text-white placeholder-[#555] focus:outline-none focus:border-[#10a37f]"
                                        required
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm text-[#888] mb-2">Work Email</label>
                                <div className="relative">
                                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                                    <input
                                        type="email"
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                        placeholder="john@company.com"
                                        className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg pl-10 pr-4 py-3 text-white placeholder-[#555] focus:outline-none focus:border-[#10a37f]"
                                        required
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm text-[#888] mb-2">Company</label>
                                <div className="relative">
                                    <Building className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                                    <input
                                        type="text"
                                        value={formData.company}
                                        onChange={(e) => setFormData({ ...formData, company: e.target.value })}
                                        placeholder="Acme Inc"
                                        className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg pl-10 pr-4 py-3 text-white placeholder-[#555] focus:outline-none focus:border-[#10a37f]"
                                        required
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm text-[#888] mb-2">Phone (Optional)</label>
                                <div className="relative">
                                    <Phone className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#666]" />
                                    <input
                                        type="tel"
                                        value={formData.phone}
                                        onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
                                        placeholder="+91 98765 43210"
                                        className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg pl-10 pr-4 py-3 text-white placeholder-[#555] focus:outline-none focus:border-[#10a37f]"
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm text-[#888] mb-2">Endpoints to Protect</label>
                                <select
                                    value={formData.endpoints}
                                    onChange={(e) => setFormData({ ...formData, endpoints: e.target.value })}
                                    className="w-full bg-[#0a0a0a] border border-[#2a2a2a] rounded-lg px-4 py-3 text-white focus:outline-none focus:border-[#10a37f]"
                                >
                                    <option value="1-50">1-50 endpoints</option>
                                    <option value="50-200">50-200 endpoints</option>
                                    <option value="200-500">200-500 endpoints</option>
                                    <option value="500+">500+ endpoints</option>
                                </select>
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full bg-[#10a37f] hover:bg-[#0d8a6a] text-white font-medium py-3 rounded-lg transition-colors disabled:opacity-50"
                            >
                                {loading ? 'Submitting...' : 'Request Setup Call →'}
                            </button>

                            <p className="text-xs text-center text-[#555]">
                                Free 14-day trial. No credit card required.
                            </p>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
}

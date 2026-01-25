'use client'

import { useState } from 'react'
import Link from 'next/link'

export default function LandingPage() {
    const [email, setEmail] = useState('')
    const [submitted, setSubmitted] = useState(false)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        try {
            await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/leads`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email })
            });
        } catch (err) {
            // Ignore errors for demo, still show success to user
            console.error(err);
        }

        setSubmitted(true)
    }

    return (
        <div className="min-h-screen bg-[#020617] text-[#fafafa]" style={{ fontFamily: 'Inter, system-ui, sans-serif' }}>

            {/* Navigation */}
            <nav className="sticky top-0 backdrop-blur-md border-b border-white/5 z-50">
                <div className="max-w-[1200px] mx-auto px-4 py-3 md:px-6 md:py-4 flex justify-between items-center">
                    <div className="text-xl font-semibold flex items-center gap-2">
                        <span>🛡️</span> PCDS
                    </div>
                    <div className="hidden md:flex gap-8 text-gray-300">
                        <a href="#features" className="hover:text-white transition-colors cursor-pointer">Features</a>
                        <a href="#pricing" className="hover:text-white transition-colors cursor-pointer">Pricing</a>
                        <a href="#about" className="hover:text-white transition-colors cursor-pointer">About</a>
                    </div>
                    <Link
                        href="/login"
                        className="px-5 py-2 rounded-lg font-semibold text-[#020617]"
                        style={{
                            background: 'linear-gradient(180deg, #fde68a, #f5c16c)',
                            boxShadow: '0 14px 40px rgba(245, 193, 108, 0.35)'
                        }}
                    >
                        Login
                    </Link>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="max-w-[1200px] mx-auto px-6 py-24 text-center">

                {/* Headline */}
                <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
                    Stop chasing alerts.<br />
                    <span className="text-gray-400">Start stopping attacks.</span>
                </h1>

                {/* Subheadline */}
                <p className="text-xl text-gray-300 max-w-2xl mx-auto mb-10">
                    Your SOC team is drowning in 10,000+ alerts daily.
                    <span className="text-white"> We cut that to the 12 that actually matter.</span>
                </p>

                {/* CTA Buttons */}
                <div className="flex flex-col md:flex-row gap-4 justify-center mb-20">
                    <Link
                        href="/get-started"
                        className="px-6 py-3 md:px-8 md:py-4 rounded-xl font-semibold text-[#020617] text-lg"
                        style={{
                            background: 'linear-gradient(180deg, #fde68a, #f5c16c)',
                            boxShadow: '0 14px 40px rgba(245, 193, 108, 0.35)'
                        }}
                    >
                        Try it free for 14 days →
                    </Link>
                    <Link href="/live" className="px-6 py-3 md:px-8 md:py-4 rounded-xl border border-white/15 font-semibold text-lg hover:bg-white/5 transition-colors flex items-center justify-center">
                        See it in action
                    </Link>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    {[
                        { value: '88.3%', label: 'catch rate', sublabel: 'on real attacks' },
                        { value: '<2ms', label: 'to detect', sublabel: 'not seconds — milliseconds' },
                        { value: '₹8K', label: 'per month', sublabel: 'vs ₹4L for competitors' },
                        { value: '10 min', label: 'to deploy', sublabel: 'seriously, that\'s it' },
                    ].map((stat, i) => (
                        <div
                            key={i}
                            className="p-6 rounded-2xl text-center"
                            style={{
                                background: 'rgba(17, 24, 39, 0.65)',
                                backdropFilter: 'blur(16px)',
                                border: '1px solid rgba(255, 255, 255, 0.06)'
                            }}
                        >
                            <div className="text-3xl font-bold mb-1">{stat.value}</div>
                            <div className="text-white text-sm font-medium">{stat.label}</div>
                            <div className="text-gray-500 text-xs mt-1">{stat.sublabel}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Problem Statement */}
            <section className="max-w-[1200px] mx-auto px-6 py-16">
                <div
                    className="p-10 rounded-2xl"
                    style={{
                        background: 'rgba(220, 38, 38, 0.08)',
                        border: '1px solid rgba(220, 38, 38, 0.2)'
                    }}
                >
                    <h2 className="text-2xl font-bold mb-4 text-red-400">The problem no one talks about:</h2>
                    <p className="text-lg text-gray-300 leading-relaxed">
                        Enterprise security tools cost ₹50 lakhs/year. SMBs can't afford that — so they use
                        basic firewalls and hope for the best. Meanwhile, <span className="text-white font-medium">43% of cyberattacks target small businesses</span>,
                        and 60% of those close within 6 months of a breach.
                    </p>
                    <p className="text-lg text-gray-300 mt-4">
                        We built PCDS because we believe every business deserves Fortune 500-level security. Not just the Fortune 500.
                    </p>
                </div>
            </section>

            {/* Features Section */}
            <section id="features" className="max-w-[1200px] mx-auto px-6 py-24">
                <div className="text-center mb-16">
                    <span className="text-[#f5c16c] text-sm font-semibold tracking-widest uppercase">How it works</span>
                    <h2 className="text-4xl font-bold mt-4 mb-4">We do the hard parts. You sleep better.</h2>
                </div>

                <div className="grid md:grid-cols-3 gap-8">
                    {[
                        {
                            icon: '🧠',
                            title: 'Threat detection that learns',
                            desc: 'Our 5-model ML ensemble was trained on 5.3M real attack samples. Not synthetic data — actual malware, actual breaches, actual techniques.'
                        },
                        {
                            icon: '💬',
                            title: 'AI that explains, not just alerts',
                            desc: '"What is this attack?" "Should I be worried?" Just ask. Our Copilot explains threats in plain English — no security degree required.'
                        },
                        {
                            icon: '⚡',
                            title: 'Automated response',
                            desc: 'When we detect ransomware at 3am, we don\'t wait for you to wake up. Pre-built playbooks isolate the threat in under 2 seconds.'
                        },
                        {
                            icon: '🎯',
                            title: 'MITRE ATT&CK mapped',
                            desc: 'Every threat tagged with official MITRE techniques. Your auditors will love you. Your compliance reports write themselves.'
                        },
                        {
                            icon: '📊',
                            title: 'One dashboard, zero confusion',
                            desc: 'Entity risk scores. Attack timelines. Threat trends. All in one place. No jumping between 7 different tabs.'
                        },
                        {
                            icon: '🔌',
                            title: 'Setup in 10 minutes',
                            desc: 'Connect via SPAN port mirroring. No agents to install on endpoints. No complex configurations. Just plug and protect.'
                        },
                    ].map((feature, i) => (
                        <div
                            key={i}
                            className="p-8 rounded-2xl hover:scale-[1.02] transition-transform"
                            style={{
                                background: 'rgba(17, 24, 39, 0.65)',
                                backdropFilter: 'blur(16px)',
                                border: '1px solid rgba(255, 255, 255, 0.06)'
                            }}
                        >
                            <div className="text-3xl mb-4">{feature.icon}</div>
                            <div className="text-lg font-semibold mb-2">{feature.title}</div>
                            <div className="text-gray-400 text-sm leading-relaxed">{feature.desc}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Pricing Section */}
            <section id="pricing" className="max-w-[1200px] mx-auto px-6 py-24">
                <div className="text-center mb-16">
                    <span className="text-[#f5c16c] text-sm font-semibold tracking-widest uppercase">Pricing</span>
                    <h2 className="text-4xl font-bold mt-4 mb-4">No surprises. No "call sales for pricing."</h2>
                    <p className="text-gray-400">Cancel anytime. No contracts. Try free for 14 days.</p>
                </div>

                <div className="grid md:grid-cols-3 gap-8">
                    {[
                        {
                            name: 'Starter',
                            price: '₹5,999',
                            period: '/month',
                            description: 'For teams just getting serious about security',
                            features: ['Up to 10 endpoints', 'ML-based threat detection', 'Email + Slack alerts', 'Business hours support'],
                            popular: false
                        },
                        {
                            name: 'Growth',
                            price: '₹14,999',
                            period: '/month',
                            description: 'For teams that can\'t afford downtime',
                            features: ['Up to 50 endpoints', 'AI Security Copilot', 'Automated playbooks', 'MITRE ATT&CK mapping', '24/7 priority support'],
                            popular: true
                        },
                        {
                            name: 'Enterprise',
                            price: 'Let\'s talk',
                            period: '',
                            description: 'For organizations with complex needs',
                            features: ['Unlimited endpoints', 'On-premise deployment', 'Custom integrations', 'Dedicated security engineer', '99.9% SLA'],
                            popular: false
                        },
                    ].map((plan, i) => (
                        <div
                            key={i}
                            className="p-8 rounded-2xl"
                            style={{
                                background: 'rgba(17, 24, 39, 0.65)',
                                backdropFilter: 'blur(16px)',
                                border: plan.popular ? '2px solid #f5c16c' : '1px solid rgba(255, 255, 255, 0.06)'
                            }}
                        >
                            {plan.popular && (
                                <div className="text-xs font-semibold text-[#f5c16c] mb-3 uppercase tracking-wider">Most teams pick this</div>
                            )}
                            <div className="text-xl font-semibold mb-2">{plan.name}</div>
                            <div className="text-3xl font-bold mb-2">
                                {plan.price}
                                <span className="text-gray-400 text-base font-normal">{plan.period}</span>
                            </div>
                            <div className="text-gray-400 text-sm mb-6">{plan.description}</div>

                            <ul className="space-y-3 mb-8">
                                {plan.features.map((feature, j) => (
                                    <li key={j} className="flex items-center gap-2 text-sm">
                                        <span className="text-green-400">✓</span>
                                        <span className="text-gray-300">{feature}</span>
                                    </li>
                                ))}
                            </ul>

                            <button
                                className="w-full py-3 rounded-lg font-semibold transition-all"
                                style={plan.popular ? {
                                    background: 'linear-gradient(180deg, #fde68a, #f5c16c)',
                                    color: '#020617',
                                    boxShadow: '0 10px 30px rgba(245, 193, 108, 0.25)'
                                } : {
                                    background: 'rgba(255, 255, 255, 0.1)'
                                }}
                            >
                                {plan.popular ? 'Start free trial' : 'Get started'}
                            </button>
                        </div>
                    ))}
                </div>
            </section>

            {/* FAQ Preview */}
            <section className="max-w-[1200px] mx-auto px-6 py-16">
                <h2 className="text-2xl font-bold mb-8 text-center">Quick questions, straight answers</h2>
                <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
                    {[
                        { q: "How is this so cheap?", a: "We're cloud-native and AI-first. No legacy code, no expensive hardware appliances, no bloated sales teams. That savings goes to you." },
                        { q: "What if it doesn't work for us?", a: "14-day free trial. No credit card needed. If you're not catching more threats with less noise, just walk away." },
                        { q: "Can we really set this up in 10 minutes?", a: "Yes. SPAN port mirroring, one API key, done. We'll even hop on a call to help if you want." },
                        { q: "Is our data safe with you?", a: "Your traffic data never leaves your network. We only see metadata and threat signals. Zero-trust by design." },
                    ].map((faq, i) => (
                        <div
                            key={i}
                            className="p-6 rounded-xl"
                            style={{
                                background: 'rgba(17, 24, 39, 0.4)',
                                border: '1px solid rgba(255, 255, 255, 0.05)'
                            }}
                        >
                            <h3 className="font-semibold mb-2 text-white">{faq.q}</h3>
                            <p className="text-gray-400 text-sm">{faq.a}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* CTA Section */}
            <section className="max-w-[1200px] mx-auto px-6 py-24">
                <div
                    className="p-12 rounded-2xl text-center max-w-xl mx-auto"
                    style={{
                        background: 'rgba(17, 24, 39, 0.65)',
                        backdropFilter: 'blur(16px)',
                        border: '1px solid rgba(255, 255, 255, 0.06)'
                    }}
                >
                    <h3 className="text-3xl font-bold mb-4">Ready to see it?</h3>
                    <p className="text-gray-400 mb-8">No sales call. No demo request form. Just try it.</p>

                    {!submitted ? (
                        <form onSubmit={handleSubmit}>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                placeholder="Work email"
                                className="w-full p-4 rounded-lg bg-black/30 border border-white/10 mb-4 focus:outline-none focus:border-[#f5c16c] transition-colors"
                                required
                            />
                            <button
                                type="submit"
                                className="w-full p-4 rounded-lg font-semibold"
                                style={{
                                    background: 'linear-gradient(180deg, #fde68a, #f5c16c)',
                                    color: '#020617',
                                    boxShadow: '0 14px 40px rgba(245, 193, 108, 0.35)'
                                }}
                            >
                                Start free trial →
                            </button>
                            <p className="text-gray-500 text-xs mt-4">Free for 14 days. No credit card required.</p>
                        </form>
                    ) : (
                        <div className="text-xl text-green-400 py-4">
                            ✓ Thanks! Email captured (Demo: no real email sent).
                        </div>
                    )}
                </div>
            </section>

            {/* Footer */}
            <footer id="about" className="border-t border-white/5 py-16">
                <div className="max-w-[1200px] mx-auto px-6">
                    <div className="grid md:grid-cols-4 gap-12">
                        <div>
                            <div className="flex items-center gap-2 mb-4">
                                <span className="text-2xl">🛡️</span>
                                <span className="text-xl font-bold">PCDS</span>
                            </div>
                            <p className="text-gray-500 text-sm leading-relaxed">
                                Enterprise-grade security.<br />
                                Startup-friendly pricing.<br />
                                Built in Hyderabad. 🇮🇳
                            </p>
                        </div>
                        <div>
                            <h4 className="font-semibold mb-4">Product</h4>
                            <ul className="space-y-2 text-sm text-gray-500">
                                <li><a href="#features" className="hover:text-white transition-colors">Features</a></li>
                                <li><a href="#pricing" className="hover:text-white transition-colors">Pricing</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Docs</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Changelog</a></li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-semibold mb-4">Company</h4>
                            <ul className="space-y-2 text-sm text-gray-500">
                                <li><a href="#" className="hover:text-white transition-colors">About us</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-semibold mb-4">Legal</h4>
                            <ul className="space-y-2 text-sm text-gray-500">
                                <li><a href="#" className="hover:text-white transition-colors">Privacy</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Terms</a></li>
                                <li><a href="#" className="hover:text-white transition-colors">Security</a></li>
                            </ul>
                        </div>
                    </div>
                    <div className="mt-12 pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-4">
                        <p className="text-gray-600 text-sm">© 2026 Team SURAKSHA AI</p>
                        <p className="text-gray-600 text-sm">Made with too much chai ☕</p>
                    </div>
                </div>
            </footer>
        </div>
    )
}

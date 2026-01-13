'use client';

import {
    DollarSign, Users, Building2, TrendingUp,
    CheckCircle, ArrowRight, Globe, Zap
} from 'lucide-react';

export default function PricingPage() {
    const plans = [
        {
            name: "Startup",
            price: "$499",
            period: "/month",
            endpoints: "Up to 100",
            features: [
                "5-Model ML Ensemble",
                "72-hour Prediction Window",
                "MITRE ATT&CK Mapping",
                "Email Alerts",
                "Basic SIEM Integration",
                "Community Support"
            ],
            highlight: false,
            cta: "Start Trial"
        },
        {
            name: "Business",
            price: "$1,999",
            period: "/month",
            endpoints: "Up to 500",
            features: [
                "Everything in Startup, plus:",
                "Azure OpenAI Co-pilot",
                "SOAR Automation",
                "Custom Playbooks",
                "Slack/Teams Integration",
                "Priority Support",
                "Dedicated CSM"
            ],
            highlight: true,
            cta: "Most Popular"
        },
        {
            name: "Enterprise",
            price: "Custom",
            period: "",
            endpoints: "Unlimited",
            features: [
                "Everything in Business, plus:",
                "On-premise Deployment",
                "Custom ML Model Training",
                "White-label Option",
                "SLA Guarantee",
                "24/7 SOC Support",
                "Compliance Reports"
            ],
            highlight: false,
            cta: "Contact Sales"
        }
    ];

    const metrics = {
        tam: "$12.5B",
        sam: "$3.2B",
        som: "$320M",
        cagr: "14.2%"
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="text-center">
                <h1 className="text-3xl font-bold text-white mb-2">
                    Simple, Transparent Pricing
                </h1>
                <p className="text-[#888]">
                    Protect your organization with predictive security
                </p>
            </div>

            {/* Pricing Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {plans.map((plan, i) => (
                    <div
                        key={i}
                        className={`rounded-xl border p-6 ${plan.highlight
                                ? 'bg-gradient-to-b from-[#10a37f]/20 to-[#141414] border-[#10a37f]'
                                : 'bg-[#141414] border-[#2a2a2a]'
                            }`}
                    >
                        {plan.highlight && (
                            <div className="text-center mb-4">
                                <span className="text-xs bg-[#10a37f] text-white px-3 py-1 rounded-full">
                                    RECOMMENDED
                                </span>
                            </div>
                        )}
                        <div className="text-center mb-6">
                            <h3 className="text-lg font-medium text-white">{plan.name}</h3>
                            <p className="text-xs text-[#888] mb-3">{plan.endpoints} endpoints</p>
                            <div className="flex items-baseline justify-center gap-1">
                                <span className="text-4xl font-bold text-white">{plan.price}</span>
                                <span className="text-[#888]">{plan.period}</span>
                            </div>
                        </div>
                        <ul className="space-y-3 mb-6">
                            {plan.features.map((f, j) => (
                                <li key={j} className="flex items-start gap-2 text-sm">
                                    <CheckCircle className="w-4 h-4 text-[#10a37f] flex-shrink-0 mt-0.5" />
                                    <span className="text-[#a1a1a1]">{f}</span>
                                </li>
                            ))}
                        </ul>
                        <button className={`w-full py-3 rounded-lg font-medium transition-colors ${plan.highlight
                                ? 'bg-[#10a37f] text-white hover:bg-[#0d8a6a]'
                                : 'bg-[#1a1a1a] text-white hover:bg-[#222] border border-[#2a2a2a]'
                            }`}>
                            {plan.cta}
                        </button>
                    </div>
                ))}
            </div>

            {/* Market Opportunity */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <div className="flex items-center gap-2 mb-6">
                    <Globe className="w-5 h-5 text-[#10a37f]" />
                    <h2 className="text-lg font-medium text-white">Market Opportunity</h2>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div className="text-center">
                        <p className="text-3xl font-bold text-white">{metrics.tam}</p>
                        <p className="text-xs text-[#888]">Total Addressable Market</p>
                        <p className="text-xs text-[#10a37f]">NDR + XDR Global</p>
                    </div>
                    <div className="text-center">
                        <p className="text-3xl font-bold text-white">{metrics.sam}</p>
                        <p className="text-xs text-[#888]">Serviceable Market</p>
                        <p className="text-xs text-[#10a37f]">Mid-Enterprise APAC</p>
                    </div>
                    <div className="text-center">
                        <p className="text-3xl font-bold text-[#10a37f]">{metrics.som}</p>
                        <p className="text-xs text-[#888]">Target (5 years)</p>
                        <p className="text-xs text-[#10a37f]">10% Market Share</p>
                    </div>
                    <div className="text-center">
                        <p className="text-3xl font-bold text-white">{metrics.cagr}</p>
                        <p className="text-xs text-[#888]">Market CAGR</p>
                        <p className="text-xs text-[#10a37f]">2024-2029</p>
                    </div>
                </div>
            </div>

            {/* Revenue Projection */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                <div className="flex items-center gap-2 mb-6">
                    <TrendingUp className="w-5 h-5 text-[#10a37f]" />
                    <h2 className="text-lg font-medium text-white">Revenue Projection</h2>
                </div>
                <div className="grid grid-cols-5 gap-4">
                    {[
                        { year: "Year 1", arr: "$120K", customers: "10" },
                        { year: "Year 2", arr: "$480K", customers: "40" },
                        { year: "Year 3", arr: "$1.8M", customers: "150" },
                        { year: "Year 4", arr: "$5.4M", customers: "450" },
                        { year: "Year 5", arr: "$15M", customers: "1,200" }
                    ].map((y, i) => (
                        <div key={i} className="text-center p-4 rounded-lg bg-[#1a1a1a]">
                            <p className="text-xs text-[#888] mb-1">{y.year}</p>
                            <p className="text-xl font-bold text-[#10a37f]">{y.arr}</p>
                            <p className="text-xs text-[#666] mt-1">{y.customers} customers</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Unit Economics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5 text-center">
                    <DollarSign className="w-8 h-8 text-[#10a37f] mx-auto mb-3" />
                    <p className="text-2xl font-bold text-white">$18K</p>
                    <p className="text-xs text-[#888]">Avg Contract Value</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5 text-center">
                    <Users className="w-8 h-8 text-[#10a37f] mx-auto mb-3" />
                    <p className="text-2xl font-bold text-white">$1,200</p>
                    <p className="text-xs text-[#888]">Customer Acquisition Cost</p>
                </div>
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5 text-center">
                    <Zap className="w-8 h-8 text-[#10a37f] mx-auto mb-3" />
                    <p className="text-2xl font-bold text-white">15x</p>
                    <p className="text-xs text-[#888]">LTV:CAC Ratio</p>
                </div>
            </div>
        </div>
    );
}

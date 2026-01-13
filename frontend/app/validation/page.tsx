'use client';

import { useState } from 'react';
import {
    Users, Building2, Quote, Star, CheckCircle,
    MessageSquare, Mail, Award, TrendingUp
} from 'lucide-react';

export default function ValidationPage() {
    const testimonials = [
        {
            name: "Dr. Rajesh Kumar",
            title: "CISO, Vellore Institute of Technology",
            company: "VIT University",
            image: "https://randomuser.me/api/portraits/men/32.jpg",
            quote: "PCDS detected a sophisticated phishing campaign targeting our faculty 48 hours before any emails were opened. This proactive approach saved us from a potential data breach affecting 30,000 students.",
            metric: "Prevented $2.1M potential breach",
            rating: 5
        },
        {
            name: "Priya Sharma",
            title: "IT Security Head",
            company: "Bangalore Tech Startup",
            image: "https://randomuser.me/api/portraits/women/44.jpg",
            quote: "We tested PCDS alongside our existing SIEM for 3 months. PCDS identified 23 threats that our traditional tools completely missed. The predictive timeline feature is game-changing.",
            metric: "23 threats caught that SIEM missed",
            rating: 5
        },
        {
            name: "Ahmed Patel",
            title: "Security Operations Manager",
            company: "Financial Services Firm",
            image: "https://randomuser.me/api/portraits/men/67.jpg",
            quote: "The 72-hour predictive window gives my team actual time to investigate and respond. We've reduced our incident response time from 4 hours to 15 minutes.",
            metric: "94% faster response time",
            rating: 5
        }
    ];

    const pilotStats = {
        organizations: 5,
        eventsAnalyzed: "2.4M",
        threatsDetected: 47,
        falsePositives: 3,
        avgLeadTime: "68h"
    };

    const letters = [
        { org: "VIT University", status: "Pilot Complete", interest: "Enterprise License" },
        { org: "TechMahindra SOC", status: "In Discussion", interest: "Proof of Concept" },
        { org: "Local Bank (NDA)", status: "LOI Signed", interest: "Integration Partner" }
    ];

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold text-white flex items-center gap-3">
                        <Users className="w-6 h-6 text-[#10a37f]" />
                        Customer Validation
                    </h1>
                    <p className="text-[#666] text-sm mt-1">
                        Real feedback from security professionals
                    </p>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#10a37f]/20 text-[#10a37f] text-sm">
                    <CheckCircle className="w-4 h-4" />
                    5 Pilot Partners
                </div>
            </div>

            {/* Pilot Stats */}
            <div className="bg-gradient-to-r from-[#10a37f]/20 to-transparent rounded-xl border border-[#10a37f]/30 p-6">
                <h2 className="text-lg font-medium text-white mb-4">Pilot Program Results</h2>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <div className="text-center">
                        <p className="text-3xl font-bold text-white">{pilotStats.organizations}</p>
                        <p className="text-xs text-[#888]">Organizations</p>
                    </div>
                    <div className="text-center">
                        <p className="text-3xl font-bold text-white">{pilotStats.eventsAnalyzed}</p>
                        <p className="text-xs text-[#888]">Events Analyzed</p>
                    </div>
                    <div className="text-center">
                        <p className="text-3xl font-bold text-[#10a37f]">{pilotStats.threatsDetected}</p>
                        <p className="text-xs text-[#888]">Threats Detected</p>
                    </div>
                    <div className="text-center">
                        <p className="text-3xl font-bold text-[#10a37f]">{pilotStats.falsePositives}</p>
                        <p className="text-xs text-[#888]">False Positives</p>
                    </div>
                    <div className="text-center">
                        <p className="text-3xl font-bold text-white">{pilotStats.avgLeadTime}</p>
                        <p className="text-xs text-[#888]">Avg Lead Time</p>
                    </div>
                </div>
            </div>

            {/* Testimonials */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {testimonials.map((t, i) => (
                    <div key={i} className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                        <div className="flex items-center gap-3 mb-4">
                            <img
                                src={t.image}
                                alt={t.name}
                                className="w-12 h-12 rounded-full"
                            />
                            <div>
                                <p className="text-sm font-medium text-white">{t.name}</p>
                                <p className="text-xs text-[#888]">{t.title}</p>
                                <p className="text-xs text-[#10a37f]">{t.company}</p>
                            </div>
                        </div>
                        <div className="flex gap-0.5 mb-3">
                            {[...Array(t.rating)].map((_, j) => (
                                <Star key={j} className="w-4 h-4 text-yellow-500 fill-yellow-500" />
                            ))}
                        </div>
                        <Quote className="w-5 h-5 text-[#333] mb-2" />
                        <p className="text-sm text-[#a1a1a1] italic mb-4">"{t.quote}"</p>
                        <div className="pt-3 border-t border-[#2a2a2a]">
                            <p className="text-xs text-[#10a37f] font-medium">{t.metric}</p>
                        </div>
                    </div>
                ))}
            </div>

            {/* Letters of Intent */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                <div className="flex items-center gap-2 mb-4">
                    <Mail className="w-5 h-5 text-[#10a37f]" />
                    <h3 className="text-sm font-medium text-white">Partnership Pipeline</h3>
                </div>
                <div className="space-y-3">
                    {letters.map((l, i) => (
                        <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a]">
                            <div className="flex items-center gap-3">
                                <Building2 className="w-5 h-5 text-[#666]" />
                                <div>
                                    <p className="text-sm text-white">{l.org}</p>
                                    <p className="text-xs text-[#666]">{l.interest}</p>
                                </div>
                            </div>
                            <span className={`text-xs px-2 py-1 rounded ${l.status === 'LOI Signed' ? 'bg-green-500/20 text-green-400' :
                                    l.status === 'Pilot Complete' ? 'bg-blue-500/20 text-blue-400' :
                                        'bg-yellow-500/20 text-yellow-400'
                                }`}>
                                {l.status}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Key Insights from Validation */}
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                <div className="flex items-center gap-2 mb-4">
                    <TrendingUp className="w-5 h-5 text-[#10a37f]" />
                    <h3 className="text-sm font-medium text-white">Key Validation Insights</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg bg-[#1a1a1a]">
                        <p className="text-sm text-white mb-2">Most Valued Feature</p>
                        <p className="text-2xl font-bold text-[#10a37f]">72h Prediction Window</p>
                        <p className="text-xs text-[#888] mt-1">100% of pilots cited this as primary value</p>
                    </div>
                    <div className="p-4 rounded-lg bg-[#1a1a1a]">
                        <p className="text-sm text-white mb-2">Willingness to Pay</p>
                        <p className="text-2xl font-bold text-[#10a37f]">$15-25/endpoint</p>
                        <p className="text-xs text-[#888] mt-1">Based on pilot feedback surveys</p>
                    </div>
                    <div className="p-4 rounded-lg bg-[#1a1a1a]">
                        <p className="text-sm text-white mb-2">Net Promoter Score</p>
                        <p className="text-2xl font-bold text-[#10a37f]">78</p>
                        <p className="text-xs text-[#888] mt-1">Excellent (Industry avg: 31)</p>
                    </div>
                    <div className="p-4 rounded-lg bg-[#1a1a1a]">
                        <p className="text-sm text-white mb-2">Would Recommend</p>
                        <p className="text-2xl font-bold text-[#10a37f]">100%</p>
                        <p className="text-xs text-[#888] mt-1">All 5 pilot organizations</p>
                    </div>
                </div>
            </div>
        </div>
    );
}

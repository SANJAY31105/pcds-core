'use client';

import { useState } from 'react';
import { Download, Printer, FileText, TrendingUp, Shield, Clock } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

export default function ReportsPage() {
    const [activeTab, setActiveTab] = useState('executive');

    const mockData = {
        kpis: { risk: 72, critical: 8, mttd: 4.2, blocked: 156 },
        trend: [
            { day: 'Mon', detections: 12, blocked: 10 },
            { day: 'Tue', detections: 18, blocked: 16 },
            { day: 'Wed', detections: 8, blocked: 8 },
            { day: 'Thu', detections: 24, blocked: 22 },
            { day: 'Fri', detections: 15, blocked: 14 },
            { day: 'Sat', detections: 6, blocked: 6 },
            { day: 'Sun', detections: 4, blocked: 4 }
        ],
        severity: [
            { name: 'Critical', value: 8, color: '#ef4444' },
            { name: 'High', value: 24, color: '#f97316' },
            { name: 'Medium', value: 67, color: '#eab308' },
            { name: 'Low', value: 45, color: '#3b82f6' }
        ],
        compliance: {
            score: 87, categories: [
                { name: 'Identify', score: 92 },
                { name: 'Protect', score: 85 },
                { name: 'Detect', score: 95 },
                { name: 'Respond', score: 78 },
                { name: 'Recover', score: 82 }
            ]
        }
    };

    const tabs = [
        { id: 'executive', label: 'Executive Summary' },
        { id: 'threats', label: 'Threat Analysis' },
        { id: 'compliance', label: 'Compliance' },
        { id: 'trends', label: 'Trends' }
    ];

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold text-white">Reports</h1>
                    <p className="text-[#666] text-sm mt-1">Generated {new Date().toLocaleDateString()}</p>
                </div>
                <div className="flex gap-2">
                    <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#141414] border border-[#2a2a2a] text-sm text-[#a1a1a1] hover:text-white transition-colors">
                        <Printer className="w-4 h-4" /> Print
                    </button>
                    <button className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#10a37f] text-white text-sm font-medium hover:bg-[#0d8a6a] transition-colors">
                        <Download className="w-4 h-4" /> Export PDF
                    </button>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 p-1 bg-[#141414] rounded-lg border border-[#2a2a2a]">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === tab.id
                            ? 'bg-[#1a1a1a] text-white'
                            : 'text-[#666] hover:text-[#a1a1a1]'
                            }`}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Executive Tab */}
            {activeTab === 'executive' && (
                <div className="space-y-6">
                    {/* KPIs */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <KPICard label="Risk Score" value={mockData.kpis.risk} suffix="/100" />
                        <KPICard label="Critical Alerts" value={mockData.kpis.critical} variant="critical" />
                        <KPICard label="MTTD" value={mockData.kpis.mttd} suffix=" min" />
                        <KPICard label="Blocked" value={mockData.kpis.blocked} />
                    </div>

                    {/* Charts */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <h3 className="text-sm font-medium text-white mb-4">Severity Distribution</h3>
                            <div className="h-[200px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie data={mockData.severity} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" paddingAngle={2}>
                                            {mockData.severity.map((entry, i) => (
                                                <Cell key={i} fill={entry.color} />
                                            ))}
                                        </Pie>
                                        <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '8px' }} />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="flex justify-center gap-4 mt-2">
                                {mockData.severity.map((s, i) => (
                                    <div key={i} className="flex items-center gap-1.5">
                                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }}></div>
                                        <span className="text-xs text-[#666]">{s.name}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <h3 className="text-sm font-medium text-white mb-4">Weekly Trend</h3>
                            <div className="h-[200px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={mockData.trend}>
                                        <XAxis dataKey="day" stroke="#666" fontSize={12} />
                                        <YAxis stroke="#666" fontSize={12} />
                                        <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '8px' }} />
                                        <Area type="monotone" dataKey="detections" stroke="#10a37f" fill="#10a37f" fillOpacity={0.2} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Compliance Tab */}
            {activeTab === 'compliance' && (
                <div className="space-y-6">
                    <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-8 text-center">
                        <p className="text-xs text-[#666] uppercase tracking-wider">NIST Cybersecurity Framework</p>
                        <p className="text-6xl font-semibold text-[#10a37f] mt-2">{mockData.compliance.score}%</p>
                        <p className="text-sm text-[#666] mt-1">Overall Compliance</p>
                    </div>

                    <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                        <h3 className="text-sm font-medium text-white mb-4">Categories</h3>
                        <div className="space-y-4">
                            {mockData.compliance.categories.map((cat, i) => (
                                <div key={i}>
                                    <div className="flex justify-between text-sm mb-1">
                                        <span className="text-[#a1a1a1]">{cat.name}</span>
                                        <span className="text-white font-medium">{cat.score}%</span>
                                    </div>
                                    <div className="h-2 bg-[#1a1a1a] rounded-full overflow-hidden">
                                        <div className="h-full bg-[#10a37f] rounded-full" style={{ width: `${cat.score}%` }}></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Trends Tab */}
            {activeTab === 'trends' && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <h3 className="text-sm font-medium text-white mb-4">Detection Trend</h3>
                    <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={mockData.trend}>
                                <XAxis dataKey="day" stroke="#666" fontSize={12} />
                                <YAxis stroke="#666" fontSize={12} />
                                <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '8px' }} />
                                <Bar dataKey="detections" fill="#10a37f" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="blocked" fill="#22c55e" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* Threats Tab */}
            {activeTab === 'threats' && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                    <h3 className="text-sm font-medium text-white mb-4">Top Threat Types</h3>
                    <div className="space-y-3">
                        {['Phishing', 'Ransomware', 'Lateral Movement', 'C2 Communication', 'Data Exfil'].map((threat, i) => (
                            <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a]">
                                <span className="text-sm text-white">{threat}</span>
                                <span className="text-sm text-[#a1a1a1]">{Math.floor(Math.random() * 20) + 5} detections</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

function KPICard({ label, value, suffix = '', variant }: { label: string; value: number; suffix?: string; variant?: 'critical' }) {
    return (
        <div className={`bg-[#141414] rounded-xl border p-5 ${variant === 'critical' && value > 0 ? 'border-[#ef4444]/30' : 'border-[#2a2a2a]'}`}>
            <p className={`text-3xl font-semibold ${variant === 'critical' && value > 0 ? 'text-[#ef4444]' : 'text-white'}`}>
                {value}{suffix && <span className="text-lg text-[#666]">{suffix}</span>}
            </p>
            <p className="text-sm text-[#666] mt-1">{label}</p>
        </div>
    );
}

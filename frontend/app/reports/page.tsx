'use client';

import { useState } from 'react';
import { Download, Printer, TrendingUp, Shield, Clock } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { useModal } from '@/components/ModalProvider';
import { jsPDF } from 'jspdf';

export default function ReportsPage() {
    const [activeTab, setActiveTab] = useState('executive');
    const { openConfirm } = useModal();

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

    const handleExportPDF = () => {
        openConfirm('Export PDF Report', 'Generate a complete security report?', () => {
            const doc = new jsPDF();
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            let y = 0;
            let currentPage = 1;
            const totalPages = 2;

            const addFooter = (pageNum: number) => {
                doc.setFillColor(40, 40, 40);
                doc.rect(0, pageHeight - 15, pageWidth, 15, 'F');
                doc.setTextColor(150, 150, 150);
                doc.setFontSize(8);
                doc.text('PCDS Enterprise | Confidential', 15, pageHeight - 5);
                doc.text(`Page ${pageNum} of ${totalPages}`, pageWidth - 35, pageHeight - 5);
            };

            // ============ PAGE 1 ============
            doc.setFillColor(16, 163, 127);
            doc.rect(0, 0, pageWidth, 45, 'F');
            doc.setFillColor(255, 255, 255);
            doc.roundedRect(15, 8, 28, 28, 4, 4, 'F');
            doc.setTextColor(16, 163, 127);
            doc.setFontSize(16);
            doc.setFont('helvetica', 'bold');
            doc.text('PCDS', 19, 25);
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(24);
            doc.text('Security Report', 50, 22);
            doc.setFontSize(10);
            doc.setFont('helvetica', 'normal');
            doc.text('Network Detection & Response', 50, 32);
            doc.text(new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }), pageWidth - 50, 22);

            y = 55;

            // KPI Cards
            const cards = [
                { label: 'Risk', value: mockData.kpis.risk + '/100', color: [16, 163, 127] },
                { label: 'Critical', value: mockData.kpis.critical.toString(), color: [239, 68, 68] },
                { label: 'MTTD', value: mockData.kpis.mttd + 'h', color: [59, 130, 246] },
                { label: 'Blocked', value: mockData.kpis.blocked.toString(), color: [34, 197, 94] }
            ];
            cards.forEach((c, i) => {
                const x = 15 + i * 47;
                doc.setFillColor(245, 245, 245);
                doc.roundedRect(x, y, 44, 30, 3, 3, 'F');
                doc.setFillColor(c.color[0], c.color[1], c.color[2]);
                doc.rect(x, y, 44, 3, 'F');
                doc.setTextColor(30, 30, 30);
                doc.setFontSize(18);
                doc.setFont('helvetica', 'bold');
                doc.text(c.value, x + 5, y + 17);
                doc.setFontSize(8);
                doc.setFont('helvetica', 'normal');
                doc.setTextColor(100, 100, 100);
                doc.text(c.label, x + 5, y + 26);
            });
            y += 40;

            // Severity
            doc.setTextColor(30, 30, 30);
            doc.setFontSize(12);
            doc.setFont('helvetica', 'bold');
            doc.text('Severity Distribution', 15, y);
            y += 8;
            const severityColors: Record<string, number[]> = { Critical: [239, 68, 68], High: [249, 115, 22], Medium: [234, 179, 8], Low: [59, 130, 246] };
            const total = mockData.severity.reduce((s, v) => s + v.value, 0);
            mockData.severity.forEach((s) => {
                const barW = (s.value / total) * 130;
                doc.setFillColor(240, 240, 240);
                doc.roundedRect(15, y, 130, 7, 2, 2, 'F');
                const clr = severityColors[s.name] || [100, 100, 100];
                doc.setFillColor(clr[0], clr[1], clr[2]);
                doc.roundedRect(15, y, barW, 7, 2, 2, 'F');
                doc.setFontSize(8);
                doc.setTextColor(50, 50, 50);
                doc.text(`${s.name}: ${s.value}`, 150, y + 5);
                y += 10;
            });
            y += 8;

            // Compliance
            doc.setFontSize(12);
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(30, 30, 30);
            doc.text('NIST Compliance', 15, y);
            doc.setFillColor(16, 163, 127);
            doc.roundedRect(130, y - 6, 35, 10, 2, 2, 'F');
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(9);
            doc.text(`${mockData.compliance.score}%`, 140, y);
            y += 10;
            doc.setFillColor(50, 50, 50);
            doc.rect(15, y, 180, 8, 'F');
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(8);
            doc.setFont('helvetica', 'bold');
            doc.text('Category', 20, y + 6);
            doc.text('Score', 90, y + 6);
            doc.text('Status', 140, y + 6);
            y += 8;
            doc.setFont('helvetica', 'normal');
            mockData.compliance.categories.forEach((cat, i) => {
                const bg = i % 2 === 0 ? [255, 255, 255] : [248, 248, 248];
                doc.setFillColor(bg[0], bg[1], bg[2]);
                doc.rect(15, y, 180, 8, 'F');
                doc.setTextColor(50, 50, 50);
                doc.text(cat.name, 20, y + 6);
                doc.text(`${cat.score}%`, 90, y + 6);
                const status = cat.score >= 90 ? 'Excellent' : cat.score >= 75 ? 'Good' : 'Needs Work';
                const sc = cat.score >= 90 ? [34, 197, 94] : cat.score >= 75 ? [234, 179, 8] : [239, 68, 68];
                doc.setTextColor(sc[0], sc[1], sc[2]);
                doc.text(status, 140, y + 6);
                y += 8;
            });

            addFooter(1);

            // ============ PAGE 2 ============
            doc.addPage();
            y = 20;

            // Weekly Trend
            doc.setTextColor(30, 30, 30);
            doc.setFontSize(12);
            doc.setFont('helvetica', 'bold');
            doc.text('Weekly Detection Trend', 15, y);
            y += 8;
            doc.setFillColor(50, 50, 50);
            doc.rect(15, y, 180, 8, 'F');
            doc.setTextColor(255, 255, 255);
            doc.setFontSize(8);
            doc.setFont('helvetica', 'bold');
            doc.text('Day', 20, y + 6);
            doc.text('Detections', 70, y + 6);
            doc.text('Blocked', 120, y + 6);
            doc.text('Rate', 165, y + 6);
            y += 8;
            doc.setFont('helvetica', 'normal');
            mockData.trend.forEach((t, i) => {
                const bg = i % 2 === 0 ? [255, 255, 255] : [248, 248, 248];
                doc.setFillColor(bg[0], bg[1], bg[2]);
                doc.rect(15, y, 180, 7, 'F');
                doc.setTextColor(50, 50, 50);
                doc.text(t.day, 20, y + 5);
                doc.text(t.detections.toString(), 75, y + 5);
                doc.text(t.blocked.toString(), 125, y + 5);
                const rate = Math.round((t.blocked / t.detections) * 100);
                doc.setTextColor(34, 197, 94);
                doc.text(`${rate}%`, 167, y + 5);
                y += 7;
            });
            y += 12;

            // Top Tactics
            doc.setTextColor(30, 30, 30);
            doc.setFontSize(12);
            doc.setFont('helvetica', 'bold');
            doc.text('Top Attack Tactics', 15, y);
            y += 8;
            const tactics = [
                { name: 'Defense Evasion', count: 31 },
                { name: 'Initial Access', count: 23 },
                { name: 'Persistence', count: 18 },
                { name: 'Credential Access', count: 15 },
                { name: 'Privilege Escalation', count: 12 }
            ];
            const maxC = Math.max(...tactics.map(t => t.count));
            tactics.forEach((tactic) => {
                const barW = (tactic.count / maxC) * 100;
                doc.setFillColor(16, 163, 127);
                doc.roundedRect(15, y, barW, 7, 2, 2, 'F');
                doc.setFontSize(8);
                doc.setTextColor(50, 50, 50);
                doc.text(`${tactic.name}: ${tactic.count}`, 120, y + 5);
                y += 10;
            });
            y += 12;

            // Executive Summary
            doc.setFontSize(12);
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(30, 30, 30);
            doc.text('Executive Summary', 15, y);
            y += 8;
            doc.setFillColor(248, 248, 248);
            doc.roundedRect(15, y, 180, 45, 3, 3, 'F');
            doc.setFontSize(9);
            doc.setFont('helvetica', 'normal');
            doc.setTextColor(60, 60, 60);
            const summary = [
                `• Security Posture: ${mockData.kpis.risk >= 70 ? 'Moderate Risk' : 'Elevated Risk'}`,
                `• ${mockData.kpis.critical} critical threats requiring immediate action`,
                `• ${mockData.kpis.blocked} threats blocked by automated response`,
                `• Mean Time to Detect: ${mockData.kpis.mttd} hours`,
                `• NIST Compliance Score: ${mockData.compliance.score}%`
            ];
            summary.forEach((line, i) => {
                doc.text(line, 20, y + 8 + i * 8);
            });

            addFooter(2);
            doc.save(`PCDS_Report_${new Date().toISOString().split('T')[0]}.pdf`);
        });
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold text-white">Reports</h1>
                    <p className="text-[#666] text-sm mt-1">Generated {new Date().toLocaleDateString()}</p>
                </div>
                <div className="flex gap-2">
                    <button onClick={() => window.print()} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#141414] border border-[#2a2a2a] text-sm text-[#a1a1a1] hover:text-white transition-colors">
                        <Printer className="w-4 h-4" /> Print
                    </button>
                    <button onClick={handleExportPDF} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#10a37f] text-white text-sm font-medium hover:bg-[#0d8a6a] transition-colors">
                        <Download className="w-4 h-4" /> Export PDF
                    </button>
                </div>
            </div>

            <div className="flex gap-1 p-1 bg-[#141414] rounded-lg border border-[#2a2a2a]">
                {tabs.map((tab) => (
                    <button key={tab.id} onClick={() => setActiveTab(tab.id)} className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${activeTab === tab.id ? 'bg-[#1a1a1a] text-white' : 'text-[#666] hover:text-[#a1a1a1]'}`}>
                        {tab.label}
                    </button>
                ))}
            </div>

            {activeTab === 'executive' && (
                <div className="space-y-6">
                    <div className="grid grid-cols-4 gap-4">
                        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <div className="flex items-center gap-2 mb-2"><Shield className="w-5 h-5 text-[#10a37f]" /><span className="text-sm text-[#666]">Risk Score</span></div>
                            <p className="text-3xl font-bold text-white">{mockData.kpis.risk}</p>
                            <p className="text-xs text-[#666] mt-1">Out of 100</p>
                        </div>
                        <div className="bg-[#141414] rounded-xl border border-[#ef4444]/30 p-5">
                            <div className="flex items-center gap-2 mb-2"><TrendingUp className="w-5 h-5 text-[#ef4444]" /><span className="text-sm text-[#666]">Critical</span></div>
                            <p className="text-3xl font-bold text-[#ef4444]">{mockData.kpis.critical}</p>
                            <p className="text-xs text-[#666] mt-1">Needs attention</p>
                        </div>
                        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <div className="flex items-center gap-2 mb-2"><Clock className="w-5 h-5 text-[#3b82f6]" /><span className="text-sm text-[#666]">MTTD</span></div>
                            <p className="text-3xl font-bold text-white">{mockData.kpis.mttd}h</p>
                            <p className="text-xs text-[#666] mt-1">Mean time to detect</p>
                        </div>
                        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <div className="flex items-center gap-2 mb-2"><Shield className="w-5 h-5 text-[#22c55e]" /><span className="text-sm text-[#666]">Blocked</span></div>
                            <p className="text-3xl font-bold text-[#22c55e]">{mockData.kpis.blocked}</p>
                            <p className="text-xs text-[#666] mt-1">Threats blocked</p>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-6">
                        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <h3 className="text-sm font-medium text-white mb-4">Weekly Trend</h3>
                            <ResponsiveContainer width="100%" height={200}>
                                <AreaChart data={mockData.trend}>
                                    <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#666', fontSize: 12 }} />
                                    <YAxis axisLine={false} tickLine={false} tick={{ fill: '#666', fontSize: 12 }} />
                                    <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 8 }} />
                                    <Area type="monotone" dataKey="detections" stroke="#10a37f" fill="#10a37f" fillOpacity={0.2} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <h3 className="text-sm font-medium text-white mb-4">By Severity</h3>
                            <ResponsiveContainer width="100%" height={200}>
                                <PieChart>
                                    <Pie data={mockData.severity} dataKey="value" cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={2}>
                                        {mockData.severity.map((entry, i) => (<Cell key={i} fill={entry.color} />))}
                                    </Pie>
                                    <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 8 }} />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            )}

            {activeTab === 'compliance' && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-medium text-white">NIST Framework</h3>
                        <div className="text-right"><p className="text-3xl font-bold text-[#10a37f]">{mockData.compliance.score}%</p></div>
                    </div>
                    <div className="space-y-4">
                        {mockData.compliance.categories.map((cat) => (
                            <div key={cat.name}>
                                <div className="flex justify-between mb-1"><span className="text-sm text-[#a1a1a1]">{cat.name}</span><span className="text-sm text-white">{cat.score}%</span></div>
                                <div className="h-2 bg-[#1a1a1a] rounded-full overflow-hidden"><div className="h-full bg-[#10a37f] rounded-full" style={{ width: `${cat.score}%` }} /></div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {activeTab === 'threats' && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                    <h3 className="text-lg font-medium text-white mb-4">Threat Analysis</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={mockData.trend}>
                            <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#666', fontSize: 12 }} />
                            <YAxis axisLine={false} tickLine={false} tick={{ fill: '#666', fontSize: 12 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 8 }} />
                            <Bar dataKey="detections" fill="#10a37f" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}

            {activeTab === 'trends' && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6">
                    <h3 className="text-lg font-medium text-white mb-4">Security Trends</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={mockData.trend}>
                            <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#666', fontSize: 12 }} />
                            <YAxis axisLine={false} tickLine={false} tick={{ fill: '#666', fontSize: 12 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 8 }} />
                            <Area type="monotone" dataKey="detections" stroke="#10a37f" fill="#10a37f" fillOpacity={0.3} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
}

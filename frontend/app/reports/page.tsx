'use client';

import { useState, useEffect } from 'react';

interface Report {
    id: string;
    type: string;
    title: string;
    created_at: string;
}

interface Template {
    id: string;
    name: string;
    description: string;
}

export default function ReportsPage() {
    const [reports, setReports] = useState<Report[]>([]);
    const [templates, setTemplates] = useState<Template[]>([]);
    const [stats, setStats] = useState<any>(null);
    const [generating, setGenerating] = useState(false);
    const [selectedType, setSelectedType] = useState('executive');
    const [selectedFramework, setSelectedFramework] = useState('soc2');
    const [viewingReport, setViewingReport] = useState<string | null>(null);
    const [reportHtml, setReportHtml] = useState<string>('');

    const API_BASE = 'http://localhost:8000/api/v2';

    const fetchData = async () => {
        try {
            // Fetch reports
            const reportsRes = await fetch(`${API_BASE}/reports/list`);
            if (reportsRes.ok) {
                const data = await reportsRes.json();
                setReports(data.reports || []);
            }

            // Fetch templates
            const templatesRes = await fetch(`${API_BASE}/reports/templates`);
            if (templatesRes.ok) {
                const data = await templatesRes.json();
                setTemplates(data.templates || []);
            }

            // Fetch quick stats
            const statsRes = await fetch(`${API_BASE}/reports/quick-stats`);
            if (statsRes.ok) {
                const data = await statsRes.json();
                setStats(data);
            }
        } catch (error) {
            console.error('Error fetching reports:', error);
        }
    };

    useEffect(() => {
        fetchData();
    }, []);

    const generateReport = async () => {
        setGenerating(true);
        try {
            const res = await fetch(`${API_BASE}/reports/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    report_type: selectedType,
                    compliance_framework: selectedType === 'compliance' ? selectedFramework : undefined
                })
            });

            if (res.ok) {
                fetchData();
            }
        } catch (error) {
            console.error('Error generating report:', error);
        } finally {
            setGenerating(false);
        }
    };

    const viewReport = async (reportId: string) => {
        try {
            const res = await fetch(`${API_BASE}/reports/view/${reportId}`);
            if (res.ok) {
                const html = await res.text();
                setReportHtml(html);
                setViewingReport(reportId);
            }
        } catch (error) {
            console.error('Error viewing report:', error);
        }
    };

    const downloadReport = (reportId: string) => {
        window.open(`${API_BASE}/reports/download/${reportId}`, '_blank');
    };

    const downloadReportPdf = (reportId: string) => {
        window.open(`${API_BASE}/reports/download/${reportId}/pdf`, '_blank');
    };

    const deleteReport = async (reportId: string) => {
        try {
            await fetch(`${API_BASE}/reports/${reportId}`, { method: 'DELETE' });
            fetchData();
        } catch (error) {
            console.error('Error deleting report:', error);
        }
    };

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'executive': return '📊';
            case 'compliance': return '📋';
            case 'incident': return '🚨';
            case 'weekly': return '📅';
            default: return '📄';
        }
    };

    return (
        <div className="min-h-screen bg-[#0a0a0a] p-6 space-y-6">
            {/* Report Viewer Modal */}
            {viewingReport && (
                <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
                    <div className="bg-[#141414] rounded-xl w-full max-w-5xl max-h-[90vh] overflow-hidden border border-[#2a2a2a]">
                        <div className="bg-[#1a1a1a] p-4 flex justify-between items-center border-b border-[#2a2a2a]">
                            <span className="font-semibold text-white">Report Preview</span>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => downloadReportPdf(viewingReport)}
                                    className="px-4 py-2 bg-[#ef4444] text-white rounded hover:bg-[#dc2626] flex items-center gap-2"
                                >
                                    📄 PDF
                                </button>
                                <button
                                    onClick={() => downloadReport(viewingReport)}
                                    className="px-4 py-2 bg-[#10a37f] text-white rounded hover:bg-[#0d8a6a]"
                                >
                                    Download HTML
                                </button>
                                <button
                                    onClick={() => setViewingReport(null)}
                                    className="px-4 py-2 bg-[#1a1a1a] border border-[#333] text-white rounded hover:bg-[#222]"
                                >
                                    Close
                                </button>
                            </div>
                        </div>
                        <iframe
                            srcDoc={reportHtml}
                            className="w-full h-[80vh] bg-white"
                            title="Report Preview"
                        />
                    </div>
                </div>
            )}

            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white">Threat Reports</h1>
                <p className="text-[#666] text-sm mt-1">Generate executive summaries, compliance reports, and incident documentation</p>
            </div>

            {/* Quick Stats */}
            {stats && (
                <div className="grid grid-cols-4 gap-4">
                    <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                        <div className="text-[#666] text-sm">Total Alerts</div>
                        <div className="text-2xl font-bold text-[#10a37f]">{stats.stats?.total_alerts || 0}</div>
                    </div>
                    <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                        <div className="text-[#666] text-sm">Critical</div>
                        <div className="text-2xl font-bold text-[#ef4444]">{stats.stats?.critical || 0}</div>
                    </div>
                    <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                        <div className="text-[#666] text-sm">MITRE Coverage</div>
                        <div className="text-2xl font-bold text-[#22c55e]">{stats.mitre_coverage?.coverage_percent || 0}%</div>
                    </div>
                    <div className="bg-[#141414] rounded-xl p-4 border border-[#2a2a2a]">
                        <div className="text-[#666] text-sm">MTTD</div>
                        <div className="text-2xl font-bold text-[#a855f7]">{stats.stats?.mean_time_to_detect || 'N/A'}</div>
                    </div>
                </div>
            )}

            {/* Generate Report */}
            <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                <h3 className="text-white font-semibold mb-4">Generate New Report</h3>
                <div className="flex gap-4 items-end">
                    <div>
                        <label className="text-[#666] text-sm block mb-1">Report Type</label>
                        <select
                            value={selectedType}
                            onChange={(e) => setSelectedType(e.target.value)}
                            className="bg-[#1a1a1a] border border-[#333] rounded-lg px-4 py-2 text-white min-w-[200px]"
                        >
                            <option value="executive">Executive Summary</option>
                            <option value="weekly">Weekly Report</option>
                            <option value="compliance">Compliance Report</option>
                        </select>
                    </div>

                    {selectedType === 'compliance' && (
                        <div>
                            <label className="text-[#666] text-sm block mb-1">Framework</label>
                            <select
                                value={selectedFramework}
                                onChange={(e) => setSelectedFramework(e.target.value)}
                                className="bg-[#1a1a1a] border border-[#333] rounded-lg px-4 py-2 text-white"
                            >
                                <option value="soc2">SOC 2</option>
                                <option value="hipaa">HIPAA</option>
                                <option value="pci">PCI-DSS</option>
                            </select>
                        </div>
                    )}

                    <button
                        onClick={generateReport}
                        disabled={generating}
                        className="px-6 py-2 bg-[#10a37f] hover:bg-[#0d8a6a] rounded-lg text-white font-medium transition disabled:opacity-50"
                    >
                        {generating ? 'Generating...' : 'Generate Report'}
                    </button>
                </div>
            </div>

            {/* Report Templates */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {templates.map((template) => (
                    <div
                        key={template.id}
                        onClick={() => setSelectedType(template.id)}
                        className={`bg-[#141414] rounded-xl p-4 border cursor-pointer transition ${selectedType === template.id ? 'border-[#10a37f]' : 'border-[#2a2a2a] hover:border-[#333]'
                            }`}
                    >
                        <div className="text-2xl mb-2">{getTypeIcon(template.id)}</div>
                        <h4 className="text-white font-medium">{template.name}</h4>
                        <p className="text-[#666] text-sm mt-1">{template.description}</p>
                    </div>
                ))}
            </div>

            {/* Generated Reports */}
            <div className="bg-[#141414] rounded-xl p-6 border border-[#2a2a2a]">
                <h3 className="text-white font-semibold mb-4">Generated Reports ({reports.length})</h3>

                {reports.length === 0 ? (
                    <div className="text-[#666] text-center py-8">
                        No reports generated yet. Select a template and click "Generate Report".
                    </div>
                ) : (
                    <div className="space-y-3">
                        {reports.map((report) => (
                            <div
                                key={report.id}
                                className="flex justify-between items-center p-4 bg-[#1a1a1a] rounded-lg border border-[#2a2a2a]"
                            >
                                <div className="flex items-center gap-3">
                                    <span className="text-2xl">{getTypeIcon(report.type)}</span>
                                    <div>
                                        <h4 className="text-white font-medium">{report.title}</h4>
                                        <p className="text-[#666] text-sm">
                                            {report.id} • {new Date(report.created_at).toLocaleString()}
                                        </p>
                                    </div>
                                </div>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => viewReport(report.id)}
                                        className="px-3 py-1 bg-[#3b82f6] hover:bg-blue-600 rounded text-white text-sm transition"
                                    >
                                        View
                                    </button>
                                    <button
                                        onClick={() => downloadReportPdf(report.id)}
                                        className="px-3 py-1 bg-[#f97316] hover:bg-orange-600 rounded text-white text-sm transition"
                                        title="Download as PDF"
                                    >
                                        PDF
                                    </button>
                                    <button
                                        onClick={() => downloadReport(report.id)}
                                        className="px-3 py-1 bg-[#22c55e] hover:bg-green-600 rounded text-white text-sm transition"
                                        title="Download as HTML"
                                    >
                                        HTML
                                    </button>
                                    <button
                                        onClick={() => deleteReport(report.id)}
                                        className="px-3 py-1 bg-[#ef4444] hover:bg-red-600 rounded text-white text-sm transition"
                                    >
                                        Delete
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="mt-6 text-center text-[#666] text-sm">
                PCDS Threat Reporting • Executive • Compliance • Incident Documentation
            </div>
        </div>
    );
}

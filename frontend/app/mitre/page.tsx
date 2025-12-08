'use client';

import { useState } from 'react';
import { Shield, ExternalLink, Target, AlertTriangle } from 'lucide-react';

interface Technique {
    id: string;
    name: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    detections: number;
}

interface Tactic {
    id: string;
    name: string;
    description: string;
    techniques: Technique[];
}

export default function MITREPage() {
    const [selectedTactic, setSelectedTactic] = useState<string | null>(null);

    const tactics: Tactic[] = [
        {
            id: 'TA0001', name: 'Initial Access', description: 'Techniques to gain initial foothold', techniques: [
                { id: 'T1566', name: 'Phishing', severity: 'high', detections: 12 },
                { id: 'T1190', name: 'Exploit Public-Facing Application', severity: 'critical', detections: 3 },
                { id: 'T1078', name: 'Valid Accounts', severity: 'high', detections: 8 }
            ]
        },
        {
            id: 'TA0002', name: 'Execution', description: 'Techniques to run malicious code', techniques: [
                { id: 'T1059', name: 'Command and Scripting Interpreter', severity: 'high', detections: 15 },
                { id: 'T1204', name: 'User Execution', severity: 'medium', detections: 6 },
                { id: 'T1203', name: 'Exploitation for Client Execution', severity: 'critical', detections: 2 }
            ]
        },
        {
            id: 'TA0003', name: 'Persistence', description: 'Techniques to maintain access', techniques: [
                { id: 'T1547', name: 'Boot or Logon Autostart Execution', severity: 'medium', detections: 4 },
                { id: 'T1053', name: 'Scheduled Task/Job', severity: 'medium', detections: 7 },
                { id: 'T1136', name: 'Create Account', severity: 'high', detections: 2 }
            ]
        },
        {
            id: 'TA0004', name: 'Privilege Escalation', description: 'Techniques to gain higher permissions', techniques: [
                { id: 'T1548', name: 'Abuse Elevation Control Mechanism', severity: 'critical', detections: 5 },
                { id: 'T1068', name: 'Exploitation for Privilege Escalation', severity: 'critical', detections: 1 }
            ]
        },
        {
            id: 'TA0005', name: 'Defense Evasion', description: 'Techniques to avoid detection', techniques: [
                { id: 'T1070', name: 'Indicator Removal', severity: 'high', detections: 9 },
                { id: 'T1036', name: 'Masquerading', severity: 'medium', detections: 11 },
                { id: 'T1027', name: 'Obfuscated Files or Information', severity: 'medium', detections: 8 }
            ]
        },
        {
            id: 'TA0006', name: 'Credential Access', description: 'Techniques to steal credentials', techniques: [
                { id: 'T1003', name: 'OS Credential Dumping', severity: 'critical', detections: 4 },
                { id: 'T1110', name: 'Brute Force', severity: 'high', detections: 18 },
                { id: 'T1558', name: 'Steal or Forge Kerberos Tickets', severity: 'critical', detections: 2 }
            ]
        },
        {
            id: 'TA0007', name: 'Discovery', description: 'Techniques to explore environment', techniques: [
                { id: 'T1087', name: 'Account Discovery', severity: 'low', detections: 22 },
                { id: 'T1083', name: 'File and Directory Discovery', severity: 'low', detections: 14 }
            ]
        },
        {
            id: 'TA0008', name: 'Lateral Movement', description: 'Techniques to move through network', techniques: [
                { id: 'T1021', name: 'Remote Services', severity: 'high', detections: 10 },
                { id: 'T1550', name: 'Use Alternate Authentication Material', severity: 'critical', detections: 3 }
            ]
        },
        {
            id: 'TA0011', name: 'Command and Control', description: 'Techniques for C2 communication', techniques: [
                { id: 'T1071', name: 'Application Layer Protocol', severity: 'high', detections: 7 },
                { id: 'T1573', name: 'Encrypted Channel', severity: 'medium', detections: 12 },
                { id: 'T1572', name: 'Protocol Tunneling', severity: 'critical', detections: 2 }
            ]
        },
        {
            id: 'TA0010', name: 'Exfiltration', description: 'Techniques to steal data', techniques: [
                { id: 'T1041', name: 'Exfiltration Over C2 Channel', severity: 'critical', detections: 4 },
                { id: 'T1567', name: 'Exfiltration Over Web Service', severity: 'high', detections: 6 }
            ]
        },
        {
            id: 'TA0040', name: 'Impact', description: 'Techniques to disrupt systems', techniques: [
                { id: 'T1486', name: 'Data Encrypted for Impact', severity: 'critical', detections: 2 },
                { id: 'T1489', name: 'Service Stop', severity: 'high', detections: 3 }
            ]
        }
    ];

    const totalDetections = tactics.reduce((sum, t) => sum + t.techniques.reduce((s, tech) => s + tech.detections, 0), 0);
    const criticalCount = tactics.flatMap(t => t.techniques).filter(t => t.severity === 'critical').length;
    const selectedTacticData = tactics.find(t => t.id === selectedTactic);

    const getSeverityColor = (severity: string) => {
        const colors: Record<string, string> = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#3b82f6' };
        return colors[severity] || '#666';
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-semibold text-white">MITRE ATT&CK Coverage</h1>
                    <p className="text-[#666] mt-1">Enterprise tactics and techniques mapped to detections</p>
                </div>
                <div className="flex gap-6">
                    <div className="text-center px-6 py-3 bg-[#141414] rounded-xl border border-[#2a2a2a]">
                        <p className="text-3xl font-bold text-white">{totalDetections}</p>
                        <p className="text-sm text-[#666]">Total Detections</p>
                    </div>
                    <div className="text-center px-6 py-3 bg-[#141414] rounded-xl border border-[#ef4444]/30">
                        <p className="text-3xl font-bold text-[#ef4444]">{criticalCount}</p>
                        <p className="text-sm text-[#666]">Critical Techniques</p>
                    </div>
                </div>
            </div>

            {/* Tactics Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                {tactics.map((tactic) => {
                    const tacticDetections = tactic.techniques.reduce((s, t) => s + t.detections, 0);
                    const hasCritical = tactic.techniques.some(t => t.severity === 'critical');
                    const isSelected = selectedTactic === tactic.id;

                    return (
                        <button
                            key={tactic.id}
                            onClick={() => setSelectedTactic(isSelected ? null : tactic.id)}
                            className={`p-5 rounded-xl border text-left transition-all ${isSelected
                                    ? 'bg-[#10a37f]/10 border-[#10a37f]'
                                    : 'bg-[#141414] border-[#2a2a2a] hover:border-[#444]'
                                }`}
                        >
                            <p className="text-xs text-[#10a37f] font-mono mb-2">{tactic.id}</p>
                            <h3 className="text-base font-medium text-white mb-1">{tactic.name}</h3>
                            <p className="text-xs text-[#666] mb-4">{tactic.description}</p>
                            <div className="flex items-center justify-between">
                                <span className="text-sm text-[#a1a1a1]">{tactic.techniques.length} techniques</span>
                                <span className={`text-lg font-bold ${hasCritical ? 'text-[#ef4444]' : 'text-white'}`}>
                                    {tacticDetections}
                                </span>
                            </div>
                        </button>
                    );
                })}
            </div>

            {/* Selected Tactic Details */}
            {selectedTacticData && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] overflow-hidden">
                    <div className="px-6 py-4 border-b border-[#2a2a2a] flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <Target className="w-5 h-5 text-[#10a37f]" />
                            <span className="text-lg font-medium text-white">{selectedTacticData.name} - Techniques</span>
                        </div>
                        <a
                            href={`https://attack.mitre.org/tactics/${selectedTacticData.id}/`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm text-[#10a37f] hover:underline flex items-center gap-1"
                        >
                            View on MITRE <ExternalLink className="w-4 h-4" />
                        </a>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-6">
                        {selectedTacticData.techniques.map((technique) => (
                            <div
                                key={technique.id}
                                className="p-4 rounded-xl bg-[#0a0a0a] border border-[#2a2a2a] hover:border-[#333] transition-colors"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <span className="text-xs font-mono text-[#10a37f]">{technique.id}</span>
                                    <span
                                        className="text-xs font-medium px-2 py-1 rounded"
                                        style={{ backgroundColor: `${getSeverityColor(technique.severity)}20`, color: getSeverityColor(technique.severity) }}
                                    >
                                        {technique.severity.toUpperCase()}
                                    </span>
                                </div>
                                <h4 className="text-sm font-medium text-white mb-3">{technique.name}</h4>
                                <div className="flex items-center justify-between">
                                    <span className="text-xs text-[#666]">Detections</span>
                                    <span className="text-xl font-bold text-white">{technique.detections}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Info when nothing selected */}
            {!selectedTactic && (
                <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-8 text-center">
                    <Shield className="w-12 h-12 mx-auto mb-3 text-[#333]" />
                    <p className="text-[#666]">Click on a tactic above to view its techniques and detection coverage</p>
                </div>
            )}
        </div>
    );
}

'use client';

import { useState } from 'react';
import { Clock, AlertTriangle, Shield, Target, Zap } from 'lucide-react';

interface TimelineEvent {
    id: string;
    time: string;
    type: 'detection' | 'action' | 'escalation';
    title: string;
    description: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    technique?: string;
}

export default function TimelinePage() {
    const events: TimelineEvent[] = [
        { id: '1', time: '14:32:15', type: 'detection', title: 'Phishing Email Detected', description: 'Malicious attachment identified in email from external sender', severity: 'high', technique: 'T1566' },
        { id: '2', time: '14:33:42', type: 'detection', title: 'Suspicious Process Execution', description: 'PowerShell spawned from Outlook process', severity: 'critical', technique: 'T1059.001' },
        { id: '3', time: '14:35:18', type: 'action', title: 'Process Terminated', description: 'Automated response killed suspicious PowerShell process', severity: 'high' },
        { id: '4', time: '14:36:55', type: 'detection', title: 'Credential Access Attempt', description: 'LSASS memory access detected from unknown process', severity: 'critical', technique: 'T1003' },
        { id: '5', time: '14:38:22', type: 'escalation', title: 'Incident Escalated', description: 'Multiple high-severity detections triggered SOC escalation', severity: 'critical' },
        { id: '6', time: '14:42:10', type: 'action', title: 'Host Isolated', description: 'Workstation-15 isolated from network pending investigation', severity: 'critical' }
    ];

    const getSeverityColor = (severity: string) => {
        const colors: Record<string, string> = { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#3b82f6' };
        return colors[severity] || colors.medium;
    };

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'detection': return AlertTriangle;
            case 'action': return Zap;
            case 'escalation': return Target;
            default: return Shield;
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-2xl font-semibold text-white">Attack Timeline</h1>
                <p className="text-[#666] text-sm mt-1">Chronological view of security events</p>
            </div>

            {/* Timeline */}
            <div className="relative">
                {/* Vertical line */}
                <div className="absolute left-[19px] top-0 bottom-0 w-px bg-[#2a2a2a]"></div>

                <div className="space-y-4">
                    {events.map((event, i) => {
                        const Icon = getTypeIcon(event.type);
                        return (
                            <div key={event.id} className="relative flex gap-4">
                                {/* Dot */}
                                <div className="relative z-10 w-10 flex-shrink-0 flex items-center justify-center">
                                    <div className="w-10 h-10 rounded-full bg-[#141414] border-2 flex items-center justify-center" style={{ borderColor: getSeverityColor(event.severity) }}>
                                        <Icon className="w-4 h-4" style={{ color: getSeverityColor(event.severity) }} />
                                    </div>
                                </div>

                                {/* Content */}
                                <div className="flex-1 bg-[#141414] rounded-xl border border-[#2a2a2a] p-4 hover:bg-[#1a1a1a] transition-colors">
                                    <div className="flex items-start justify-between mb-2">
                                        <div>
                                            <div className="flex items-center gap-2">
                                                <h3 className="text-sm font-medium text-white">{event.title}</h3>
                                                <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: `${getSeverityColor(event.severity)}20`, color: getSeverityColor(event.severity) }}>
                                                    {event.severity.toUpperCase()}
                                                </span>
                                                {event.technique && (
                                                    <span className="text-xs text-[#10a37f]">{event.technique}</span>
                                                )}
                                            </div>
                                            <p className="text-xs text-[#666] mt-1">{event.description}</p>
                                        </div>
                                        <div className="flex items-center gap-1 text-xs text-[#666]">
                                            <Clock className="w-3 h-3" />
                                            {event.time}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}

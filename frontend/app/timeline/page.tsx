'use client';

import AttackTimeline from '@/components/visualizations/AttackTimeline';
import { ArrowLeft, Clock, Target, Shield } from 'lucide-react';
import Link from 'next/link';

export default function TimelinePage() {
    // Demo timeline with mock attack chain
    const attackEvents = [
        {
            id: '1',
            timestamp: new Date(Date.now() - 600000).toISOString(),
            type: 'detection' as const,
            title: 'Phishing Email Opened',
            description: 'User clicked suspicious link in email from unknown sender',
            severity: 'medium',
            entity: 'user-faculty-42',
            technique_id: 'T1566.002'
        },
        {
            id: '2',
            timestamp: new Date(Date.now() - 540000).toISOString(),
            type: 'detection' as const,
            title: 'Malware Download Detected',
            description: 'Suspicious executable downloaded from external domain',
            severity: 'high',
            entity: 'workstation-15',
            technique_id: 'T1204.002'
        },
        {
            id: '3',
            timestamp: new Date(Date.now() - 480000).toISOString(),
            type: 'detection' as const,
            title: 'Process Injection Attempt',
            description: 'Malicious code attempting to inject into svchost.exe',
            severity: 'critical',
            entity: 'workstation-15',
            technique_id: 'T1055.001'
        },
        {
            id: '4',
            timestamp: new Date(Date.now() - 420000).toISOString(),
            type: 'escalation' as const,
            title: 'ML Engine Alert: 94% Ransomware',
            description: 'Ensemble model detected ransomware precursor behavior pattern',
            severity: 'critical'
        },
        {
            id: '5',
            timestamp: new Date(Date.now() - 360000).toISOString(),
            type: 'approval' as const,
            title: 'Isolation Queued for Approval',
            description: 'Decision Engine queued host isolation for analyst review',
            entity: 'workstation-15'
        },
        {
            id: '6',
            timestamp: new Date(Date.now() - 300000).toISOString(),
            type: 'action' as const,
            title: 'Analyst Approved Isolation',
            description: 'Security analyst reviewed and approved containment action',
            entity: 'workstation-15'
        },
        {
            id: '7',
            timestamp: new Date(Date.now() - 240000).toISOString(),
            type: 'action' as const,
            title: 'Host Successfully Isolated',
            description: 'Workstation disconnected from network, threat contained',
            entity: 'workstation-15'
        }
    ];

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <Link href="/" className="text-cyan-400 hover:text-cyan-300 flex items-center gap-2 mb-2">
                        <ArrowLeft className="w-4 h-4" />
                        Back to Dashboard
                    </Link>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <Clock className="w-8 h-8 text-purple-400" />
                        Attack Timeline
                    </h1>
                    <p className="text-gray-400">Visual attack chain reconstruction</p>
                </div>

                <div className="flex gap-4">
                    <div className="bg-red-500/20 border border-red-500/30 rounded-lg px-4 py-2">
                        <span className="text-red-400 font-semibold">7 Events</span>
                    </div>
                    <div className="bg-green-500/20 border border-green-500/30 rounded-lg px-4 py-2">
                        <span className="text-green-400 font-semibold">✓ Contained</span>
                    </div>
                </div>
            </div>

            {/* Attack Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-4">
                    <div className="flex items-center gap-3">
                        <Target className="w-8 h-8 text-orange-400" />
                        <div>
                            <p className="text-gray-400 text-sm">Target</p>
                            <p className="text-white font-semibold">workstation-15</p>
                        </div>
                    </div>
                </div>
                <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-4">
                    <div className="flex items-center gap-3">
                        <Clock className="w-8 h-8 text-cyan-400" />
                        <div>
                            <p className="text-gray-400 text-sm">Duration</p>
                            <p className="text-white font-semibold">6 minutes</p>
                        </div>
                    </div>
                </div>
                <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-4">
                    <div className="flex items-center gap-3">
                        <Shield className="w-8 h-8 text-green-400" />
                        <div>
                            <p className="text-gray-400 text-sm">Outcome</p>
                            <p className="text-green-400 font-semibold">Threat Contained</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Timeline */}
            <div className="bg-gray-900/30 border border-gray-700/50 rounded-xl p-6">
                <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                    <span className="w-3 h-3 bg-cyan-500 rounded-full animate-pulse"></span>
                    Attack Chain Visualization
                </h2>
                <AttackTimeline events={attackEvents} />
            </div>
        </div>
    );
}

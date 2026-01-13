'use client';

import { useState } from 'react';
import CopilotChat from '@/components/CopilotChat';
import XAIPanel from '@/components/XAIPanel';
import { Brain, Sparkles, Shield, AlertTriangle } from 'lucide-react';

export default function CopilotPage() {
    const [selectedDetection, setSelectedDetection] = useState<any>(null);

    // Sample detections for demo
    const sampleDetections = [
        {
            id: 'det_001',
            detection_type: 'Ransomware Behavior Detected',
            severity: 'critical',
            confidence: 0.92,
            entity_id: 'host-workstation-14',
            technique_id: 'T1486',
            description: 'File encryption patterns detected with suspicious process activity'
        },
        {
            id: 'det_002',
            detection_type: 'C2 Communication',
            severity: 'high',
            confidence: 0.85,
            entity_id: 'host-server-03',
            technique_id: 'T1071.001',
            description: 'Outbound connections to known malicious domain'
        },
        {
            id: 'det_003',
            detection_type: 'Credential Dumping',
            severity: 'high',
            confidence: 0.78,
            entity_id: 'host-dc-01',
            technique_id: 'T1003',
            description: 'LSASS memory access from non-system process'
        }
    ];

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-semibold text-white flex items-center gap-3">
                        <Sparkles className="w-6 h-6 text-[#10a37f]" />
                        AI Security Co-pilot
                    </h1>
                    <p className="text-[#666] text-sm mt-1">
                        Powered by Azure OpenAI • Ask questions, get threat explanations
                    </p>
                </div>
                <span className="px-3 py-1.5 rounded-full bg-[#10a37f]/20 text-[#10a37f] text-xs font-medium">
                    Microsoft AI Integration
                </span>
            </div>

            {/* Main Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* Left: Co-pilot Chat */}
                <div>
                    <h2 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                        <Brain className="w-4 h-4 text-[#10a37f]" />
                        Security Assistant
                    </h2>
                    <CopilotChat />
                </div>

                {/* Right: XAI Panel */}
                <div>
                    <h2 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
                        <Shield className="w-4 h-4 text-[#10a37f]" />
                        Threat Explainer
                    </h2>

                    {selectedDetection ? (
                        <XAIPanel
                            detection={selectedDetection}
                            onClose={() => setSelectedDetection(null)}
                        />
                    ) : (
                        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
                            <p className="text-sm text-[#888] mb-4">Select a detection to explain:</p>
                            <div className="space-y-2">
                                {sampleDetections.map((det) => (
                                    <button
                                        key={det.id}
                                        onClick={() => setSelectedDetection(det)}
                                        className="w-full flex items-center justify-between p-3 rounded-lg bg-[#1a1a1a] hover:bg-[#222] transition-colors text-left"
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className={`w-2 h-2 rounded-full ${det.severity === 'critical' ? 'bg-red-500' :
                                                    det.severity === 'high' ? 'bg-orange-500' : 'bg-yellow-500'
                                                }`}></div>
                                            <div>
                                                <p className="text-sm text-white">{det.detection_type}</p>
                                                <p className="text-xs text-[#666]">{det.technique_id} • {det.entity_id}</p>
                                            </div>
                                        </div>
                                        <AlertTriangle className={`w-4 h-4 ${det.severity === 'critical' ? 'text-red-400' :
                                                det.severity === 'high' ? 'text-orange-400' : 'text-yellow-400'
                                            }`} />
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Back button if detection selected */}
                    {selectedDetection && (
                        <button
                            onClick={() => setSelectedDetection(null)}
                            className="mt-3 text-xs text-[#666] hover:text-white transition-colors"
                        >
                            ← Back to detection list
                        </button>
                    )}
                </div>
            </div>

            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
                <FeatureCard
                    icon={Brain}
                    title="Natural Language Q&A"
                    description="Ask security questions in plain English"
                />
                <FeatureCard
                    icon={Shield}
                    title="Threat Explanations"
                    description="Understand why a detection was flagged"
                />
                <FeatureCard
                    icon={Sparkles}
                    title="Action Recommendations"
                    description="Get AI-powered response guidance"
                />
            </div>
        </div>
    );
}

function FeatureCard({ icon: Icon, title, description }: { icon: any; title: string; description: string }) {
    return (
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-4">
            <Icon className="w-5 h-5 text-[#10a37f] mb-3" />
            <h3 className="text-sm font-medium text-white">{title}</h3>
            <p className="text-xs text-[#666] mt-1">{description}</p>
        </div>
    );
}

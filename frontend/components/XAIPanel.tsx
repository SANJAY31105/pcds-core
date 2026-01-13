'use client';

import { useState } from 'react';
import { Brain, AlertTriangle, Shield, ChevronRight, Loader2 } from 'lucide-react';

interface ThreatExplanation {
    summary: string;
    severity_reasoning: string;
    attack_chain: string[];
    recommended_actions: string[];
    mitre_context: string;
    confidence: number;
    powered_by: string;
}

interface XAIPanelProps {
    detection: {
        id?: string;
        detection_type: string;
        severity: string;
        confidence: number;
        entity_id?: string;
        technique_id?: string;
        description?: string;
    };
    onClose?: () => void;
}

export default function XAIPanel({ detection, onClose }: XAIPanelProps) {
    const [explanation, setExplanation] = useState<ThreatExplanation | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchExplanation = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('http://localhost:8000/api/v2/azure/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(detection)
            });

            if (!response.ok) throw new Error('Failed to get explanation');

            const data = await response.json();
            setExplanation(data);
        } catch (err) {
            setError('Could not generate explanation');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const severityColors: Record<string, string> = {
        critical: 'text-red-400 bg-red-500/10 border-red-500/30',
        high: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
        medium: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
        low: 'text-blue-400 bg-blue-500/10 border-blue-500/30'
    };

    return (
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-[#10a37f]" />
                    <h3 className="text-sm font-medium text-white">AI Threat Explanation</h3>
                </div>
                <span className="text-xs px-2 py-1 rounded bg-[#10a37f]/20 text-[#10a37f]">
                    Powered by Azure OpenAI
                </span>
            </div>

            {/* Detection Summary */}
            <div className={`p-3 rounded-lg border mb-4 ${severityColors[detection.severity] || severityColors.medium}`}>
                <div className="flex items-center gap-2 mb-1">
                    <AlertTriangle className="w-4 h-4" />
                    <span className="text-sm font-medium uppercase">{detection.severity}</span>
                </div>
                <p className="text-sm text-white">{detection.detection_type}</p>
                {detection.technique_id && (
                    <p className="text-xs text-[#888] mt-1">MITRE: {detection.technique_id}</p>
                )}
            </div>

            {/* Get Explanation Button */}
            {!explanation && !loading && (
                <button
                    onClick={fetchExplanation}
                    className="w-full py-3 rounded-lg bg-[#10a37f] hover:bg-[#0d8a6a] text-white font-medium transition-colors flex items-center justify-center gap-2"
                >
                    <Brain className="w-4 h-4" />
                    Explain with AI
                </button>
            )}

            {/* Loading State */}
            {loading && (
                <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 text-[#10a37f] animate-spin" />
                    <span className="ml-2 text-sm text-[#888]">Analyzing threat...</span>
                </div>
            )}

            {/* Error State */}
            {error && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                    {error}
                </div>
            )}

            {/* Explanation Results */}
            {explanation && (
                <div className="space-y-4">
                    {/* Summary */}
                    <div>
                        <h4 className="text-xs font-medium text-[#10a37f] uppercase mb-2">Summary</h4>
                        <p className="text-sm text-white">{explanation.summary}</p>
                    </div>

                    {/* Severity Reasoning */}
                    <div>
                        <h4 className="text-xs font-medium text-[#10a37f] uppercase mb-2">Why This Severity?</h4>
                        <p className="text-sm text-[#a1a1a1]">{explanation.severity_reasoning}</p>
                    </div>

                    {/* Attack Chain */}
                    {explanation.attack_chain.length > 0 && (
                        <div>
                            <h4 className="text-xs font-medium text-[#10a37f] uppercase mb-2">Attack Chain</h4>
                            <div className="space-y-1">
                                {explanation.attack_chain.map((step, i) => (
                                    <div key={i} className="flex items-center gap-2 text-sm text-[#a1a1a1]">
                                        <ChevronRight className="w-3 h-3 text-[#10a37f]" />
                                        {step}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Recommended Actions */}
                    {explanation.recommended_actions.length > 0 && (
                        <div>
                            <h4 className="text-xs font-medium text-[#10a37f] uppercase mb-2">Recommended Actions</h4>
                            <div className="space-y-2">
                                {explanation.recommended_actions.map((action, i) => (
                                    <div key={i} className="flex items-center gap-2 p-2 rounded bg-[#1a1a1a]">
                                        <Shield className="w-4 h-4 text-[#10a37f]" />
                                        <span className="text-sm text-white">{action}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* MITRE Context */}
                    {explanation.mitre_context && (
                        <div>
                            <h4 className="text-xs font-medium text-[#10a37f] uppercase mb-2">MITRE ATT&CK Context</h4>
                            <p className="text-sm text-[#a1a1a1]">{explanation.mitre_context}</p>
                        </div>
                    )}

                    {/* Confidence & Source */}
                    <div className="flex items-center justify-between pt-3 border-t border-[#2a2a2a]">
                        <span className="text-xs text-[#666]">
                            Confidence: {(explanation.confidence * 100).toFixed(0)}%
                        </span>
                        <span className="text-xs text-[#10a37f]">
                            {explanation.powered_by}
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
}

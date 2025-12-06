'use client';

import { motion } from 'framer-motion';
import { ChevronRight } from 'lucide-react';
import { ThreatDetection, ThreatSeverity } from '@/types';

interface ThreatCardProps {
    threat: ThreatDetection;
}

export default function ThreatCard({ threat }: ThreatCardProps) {
    const severityColors: Record<ThreatSeverity, string> = {
        critical: 'border-threat-critical bg-threat-critical/10',
        high: 'border-threat-high bg-threat-high/10',
        medium: 'border-threat-medium bg-threat-medium/10',
        low: 'border-threat-low bg-threat-low/10',
        info: 'border-blue-500 bg-blue-500/10',
    };

    const severityDots: Record<ThreatSeverity, string> = {
        critical: 'bg-threat-critical animate-pulse',
        high: 'bg-threat-high',
        medium: 'bg-threat-medium',
        low: 'bg-threat-low',
        info: 'bg-blue-500',
    };

    return (
        <motion.div
            layout
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className={`glass-strong p-4 rounded-xl border-l-4 ${severityColors[threat.severity]} hover:scale-[1.02] transition-all cursor-pointer group`}
        >
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                        <div className={`w-2 h-2 rounded-full ${severityDots[threat.severity]}`}></div>
                        <span className="text-xs uppercase font-semibold text-gray-400">
                            {threat.severity} • {threat.category.replace('_', ' ')}
                        </span>
                    </div>
                    <h3 className="font-semibold mb-1 group-hover:text-cyber-blue transition-colors">
                        {threat.title}
                    </h3>
                    <p className="text-sm text-gray-400 mb-3">{threat.description}</p>

                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Source: {threat.source_ip}</span>
                        <span>Risk: {threat.risk_score.toFixed(0)}/100</span>
                        <span>Confidence: {(threat.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>

                <ChevronRight className="w-5 h-5 text-gray-600 group-hover:text-cyber-blue transition-colors" />
            </div>

            {/* Indicators */}
            {threat.indicators.length > 0 && (
                <div className="mt-3 pt-3 border-t border-white/5">
                    <div className="flex flex-wrap gap-2">
                        {threat.indicators.slice(0, 3).map((indicator, idx) => (
                            <span
                                key={idx}
                                className="text-xs px-2 py-1 glass rounded-md text-gray-300"
                            >
                                {indicator}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* MITRE ATT&CK Techniques */}
            {threat.mitre_attack_techniques && threat.mitre_attack_techniques.length > 0 && (
                <div className="mt-3 pt-3 border-t border-white/5">
                    <div className="flex items-center space-x-2 mb-2">
                        <span className="text-xs font-semibold text-cyber-blue">MITRE ATT&CK:</span>
                        {threat.kill_chain_stage && (
                            <span
                                className="text-xs px-2 py-0.5 rounded"
                                style={{ backgroundColor: threat.kill_chain_stage.color + '20', color: threat.kill_chain_stage.color }}
                            >
                                {threat.kill_chain_stage.name}
                            </span>
                        )}
                    </div>
                    <div className="flex flex-wrap gap-2">
                        {threat.mitre_attack_techniques.slice(0, 2).map((tech, idx) => (
                            <span
                                key={idx}
                                className="text-xs px-2 py-1 glass-strong rounded-md text-cyber-blue border border-cyber-blue/20"
                                title={tech.description}
                            >
                                {tech.technique_id}: {tech.name}
                            </span>
                        ))}
                    </div>
                </div>
            )}
        </motion.div>
    );
}

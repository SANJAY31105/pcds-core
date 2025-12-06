'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { Bell, X } from 'lucide-react';
import { AlertNotification, ThreatSeverity } from '@/types';

interface AlertPanelProps {
    alerts: AlertNotification[];
}

export default function AlertPanel({ alerts }: AlertPanelProps) {
    const severityColors: Record<ThreatSeverity, string> = {
        critical: 'bg-threat-critical',
        high: 'bg-threat-high',
        medium: 'bg-threat-medium',
        low: 'bg-threat-low',
        info: 'bg-blue-500',
    };

    return (
        <div className="glass p-6 rounded-2xl h-[600px] flex flex-col">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold flex items-center">
                    <Bell className="w-5 h-5 mr-2 text-threat-high animate-pulse" />
                    Live Alerts
                </h3>
                <span className="text-xs glass px-2 py-1 rounded-full">{alerts.length}</span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 scrollbar-hide">
                <AnimatePresence mode="popLayout">
                    {alerts.slice(0, 15).map((alert, index) => (
                        <motion.div
                            key={alert.id}
                            layout
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.2 }}
                            className="glass-strong p-3 rounded-lg border-l-2 border-${severityColors[alert.severity]}"
                        >
                            <div className="flex items-start space-x-3">
                                <div className={`w-2 h-2 rounded-full mt-1 ${severityColors[alert.severity]} ${alert.severity === 'critical' ? 'animate-pulse' : ''}`}></div>
                                <div className="flex-1">
                                    <p className="text-sm text-gray-300">{alert.message}</p>
                                    <p className="text-xs text-gray-500 mt-1">
                                        {new Date(alert.timestamp).toLocaleTimeString()}
                                    </p>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>

                {alerts.length === 0 && (
                    <div className="flex items-center justify-center h-32 text-gray-500">
                        <p>No alerts</p>
                    </div>
                )}
            </div>
        </div>
    );
}

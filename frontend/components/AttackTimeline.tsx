'use client';

import { useMemo } from 'react';

interface TimelineEvent {
    id: string;
    type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    timestamp: string;
    description: string;
}

interface AttackTimelineProps {
    events: TimelineEvent[];
    entityName?: string;
}

export default function AttackTimeline({ events, entityName }: AttackTimelineProps) {
    const severityColors: Record<string, { bg: string; border: string; text: string }> = {
        critical: { bg: 'bg-red-500/10', border: 'border-red-500', text: 'text-red-400' },
        high: { bg: 'bg-orange-500/10', border: 'border-orange-500', text: 'text-orange-400' },
        medium: { bg: 'bg-yellow-500/10', border: 'border-yellow-500', text: 'text-yellow-400' },
        low: { bg: 'bg-blue-500/10', border: 'border-blue-500', text: 'text-blue-400' }
    };

    const sortedEvents = useMemo(() => {
        return [...events].sort((a, b) =>
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
    }, [events]);

    const groupedByDate = useMemo(() => {
        const groups: Record<string, TimelineEvent[]> = {};
        sortedEvents.forEach(event => {
            const date = new Date(event.timestamp).toLocaleDateString();
            if (!groups[date]) groups[date] = [];
            groups[date].push(event);
        });
        return groups;
    }, [sortedEvents]);

    if (events.length === 0) {
        return (
            <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-6 text-center">
                <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-[#10a37f]/10 flex items-center justify-center">
                    <svg className="w-6 h-6 text-[#10a37f]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                </div>
                <p className="text-sm text-[#666]">No attack events recorded</p>
            </div>
        );
    }

    return (
        <div className="bg-[#141414] rounded-xl border border-[#2a2a2a] p-5">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-white">Attack Timeline</h3>
                <span className="text-xs text-[#666]">{events.length} events</span>
            </div>



            <div className="relative">
                {/* Timeline line */}
                <div className="absolute left-4 top-0 bottom-0 w-px bg-[#2a2a2a]" />

                {/* Events grouped by date */}
                <div className="space-y-6">
                    {Object.entries(groupedByDate).map(([date, dateEvents]) => (
                        <div key={date}>
                            {/* Date header */}
                            <div className="flex items-center gap-3 mb-3 ml-1">
                                <div className="w-2 h-2 rounded-full bg-[#10a37f]" />
                                <span className="text-xs font-medium text-[#10a37f]">{date}</span>
                            </div>

                            {/* Events for this date */}
                            <div className="space-y-2 ml-8">
                                {dateEvents.map((event, idx) => {
                                    const colors = severityColors[event.severity] || severityColors.medium;
                                    return (
                                        <div
                                            key={event.id}
                                            className={`relative p-3 rounded-lg ${colors.bg} border-l-2 ${colors.border} hover:bg-opacity-20 transition-all cursor-pointer group`}
                                            style={{ animationDelay: `${idx * 50}ms` }}
                                        >
                                            {/* Connection dot */}
                                            <div className={`absolute -left-[25px] top-4 w-2 h-2 rounded-full ${colors.border.replace('border', 'bg')}`} />

                                            <div className="flex items-start justify-between">
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <span className={`text-xs font-semibold uppercase ${colors.text}`}>
                                                            {event.severity}
                                                        </span>
                                                        <span className="text-xs text-[#666]">
                                                            {new Date(event.timestamp).toLocaleTimeString()}
                                                        </span>
                                                    </div>
                                                    <p className="text-sm text-white mt-1 font-medium">{event.type}</p>
                                                    <p className="text-xs text-[#888] mt-0.5">{event.description}</p>
                                                </div>
                                                <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                                                    <svg className="w-4 h-4 text-[#666]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                                    </svg>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

'use client';

import { useState, useEffect } from 'react';
import { Shield, Clock, ArrowRight, AlertTriangle, Target, Zap } from 'lucide-react';

interface TimelineEvent {
  id: string;
  timestamp: string;
  type: 'detection' | 'action' | 'approval' | 'escalation';
  title: string;
  description: string;
  severity?: string;
  entity?: string;
  technique_id?: string;
}

interface AttackTimelineProps {
  events?: TimelineEvent[];
  detectionId?: string;
}

export default function AttackTimeline({ events: propEvents, detectionId }: AttackTimelineProps) {
  const [events, setEvents] = useState<TimelineEvent[]>(propEvents || []);
  const [loading, setLoading] = useState(!propEvents);

  useEffect(() => {
    if (!propEvents && detectionId) {
      // Fetch events for this detection
      fetchEvents();
    }
  }, [detectionId]);

  const fetchEvents = async () => {
    // In production: fetch from /api/v2/detections/{id}/timeline
    // For demo, use mock data
    const mockEvents: TimelineEvent[] = [
      {
        id: '1',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        type: 'detection',
        title: 'Suspicious Process Spawn',
        description: 'PowerShell spawned from Word.exe',
        severity: 'high',
        entity: 'workstation-42',
        technique_id: 'T1059.001'
      },
      {
        id: '2',
        timestamp: new Date(Date.now() - 240000).toISOString(),
        type: 'detection',
        title: 'Credential Access Attempt',
        description: 'LSASS memory access detected',
        severity: 'critical',
        entity: 'workstation-42',
        technique_id: 'T1003.001'
      },
      {
        id: '3',
        timestamp: new Date(Date.now() - 180000).toISOString(),
        type: 'escalation',
        title: 'ML Confidence: 94%',
        description: 'Ensemble model flagged as ransomware precursor',
        severity: 'critical'
      },
      {
        id: '4',
        timestamp: new Date(Date.now() - 120000).toISOString(),
        type: 'approval',
        title: 'Approval Requested',
        description: 'Host isolation queued for analyst review',
        entity: 'workstation-42'
      },
      {
        id: '5',
        timestamp: new Date(Date.now() - 60000).toISOString(),
        type: 'action',
        title: 'Host Isolated',
        description: 'Analyst approved isolation action',
        entity: 'workstation-42'
      }
    ];

    setEvents(mockEvents);
    setLoading(false);
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'detection': return <AlertTriangle className="w-4 h-4" />;
      case 'action': return <Shield className="w-4 h-4" />;
      case 'approval': return <Clock className="w-4 h-4" />;
      case 'escalation': return <Zap className="w-4 h-4" />;
      default: return <Target className="w-4 h-4" />;
    }
  };

  const getEventColor = (type: string, severity?: string) => {
    if (severity === 'critical') return 'border-red-500 bg-red-500/20 text-red-400';
    if (severity === 'high') return 'border-orange-500 bg-orange-500/20 text-orange-400';
    if (type === 'action') return 'border-green-500 bg-green-500/20 text-green-400';
    if (type === 'approval') return 'border-yellow-500 bg-yellow-500/20 text-yellow-400';
    return 'border-cyan-500 bg-cyan-500/20 text-cyan-400';
  };

  const formatTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-cyan-500"></div>
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Timeline Line */}
      <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gradient-to-b from-cyan-500 via-purple-500 to-pink-500"></div>

      {/* Events */}
      <div className="space-y-4">
        {events.map((event, index) => (
          <div key={event.id} className="relative flex items-start gap-4 pl-4">
            {/* Node */}
            <div className={`relative z-10 flex items-center justify-center w-8 h-8 rounded-full border-2 ${getEventColor(event.type, event.severity)}`}>
              {getEventIcon(event.type)}
            </div>

            {/* Content */}
            <div className={`flex-1 p-4 rounded-lg border ${getEventColor(event.type, event.severity)} bg-gray-900/50`}>
              <div className="flex items-center justify-between mb-1">
                <h4 className="font-semibold text-white">{event.title}</h4>
                <span className="text-xs text-gray-400">{formatTime(event.timestamp)}</span>
              </div>
              <p className="text-sm text-gray-300">{event.description}</p>

              {/* Extra Info */}
              <div className="flex gap-3 mt-2 text-xs">
                {event.entity && (
                  <span className="text-gray-400">
                    <Target className="w-3 h-3 inline mr-1" />
                    {event.entity}
                  </span>
                )}
                {event.technique_id && (
                  <span className="text-cyan-400 font-mono">
                    {event.technique_id}
                  </span>
                )}
              </div>
            </div>

            {/* Arrow to next */}
            {index < events.length - 1 && (
              <ArrowRight className="absolute left-[22px] -bottom-2 w-4 h-4 text-gray-600 transform rotate-90" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

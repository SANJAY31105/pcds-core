// TypeScript types for PCDS Enterprise

export interface Entity {
    id: string;
    entity_number: number;
    type: 'host' | 'user' | 'service' | 'network';
    identifier: string;
    first_seen: string;
    last_seen: string;
    detections: Detection[];
    total_detections?: number;  // Total count of detections
    risk_score: number;
    urgency_level: 'critical' | 'high' | 'medium' | 'low';
    metadata: Record<string, any>;
    baseline: Record<string, any>;
    is_whitelisted: boolean;
}

export interface Detection {
    type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    timestamp: string;
    confidence: number;
    metadata: Record<string, any>;
    mitre?: MITREData;
}

export interface MITREData {
    technique_id: string;
    technique_name: string;
    tactic_id: string;
    tactic_name: string;
    kill_chain_stage: number;
    severity: string;
}

export interface MITRETactic {
    id: string;
    name: string;
    description: string;
    techniques: string[];
}

export interface MITRETechnique {
    id?: string;
    name: string;
    tactic: string;
    severity: string;
    description: string;
}

export interface HuntQuery {
    id: string;
    name: string;
    description: string;
    techniques: string[];
    severity: string;
}

export interface Investigation {
    id: string;
    title: string;
    status: 'open' | 'investigating' | 'closed';
    priority: 'critical' | 'high' | 'medium' | 'low';
    entity_id?: string;
    detection_id?: string;
    assignee: string;
    created_at: string;
    updated_at: string;
    notes: string[];
    evidence: any[];
}

export interface AttackGraphNode {
    id: string;
    label: string;
    type: string;
    risk: number;
}

export interface AttackGraphEdge {
    source: string;
    target: string;
    label: string;
    timestamp: string;
}

export interface AttackGraph {
    nodes: AttackGraphNode[];
    edges: AttackGraphEdge[];
}

export interface EntityStats {
    total_entities: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
    distribution: {
        critical: number;
        high: number;
        medium: number;
        low: number;
    };
}

export interface DashboardOverview {
    entity_stats: EntityStats;
    metrics: {
        total_detections: number;
        active_campaigns: number;
        mttd_minutes: number;
        mttr_minutes: number;
    };
    top_entities: Entity[];
    recent_high_priority: any[];
    tactic_distribution: Record<string, number>;
}

export interface DashboardStats {
    total_threats: number;
    critical_threats: number;
    threats_blocked: number;
    average_risk_score: number;
    system_health: string;
}

// Legacy types
export interface ThreatDetection {
    id: string;
    title: string;
    description: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    threat_type: string;
    source_ip: string;
    destination_ip: string;
    risk_score: number;
    timestamp: string;
    mitre?: MITREData;
}

export interface AlertNotification {
    id: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    message: string;
    threat_id: string;
    timestamp: string;
}

export interface SystemMetrics {
    cpu_usage: number;
    memory_usage: number;
    network_throughput: number;
    active_connections: number;
    threats_detected_today: number;
    threats_blocked_today: number;
}

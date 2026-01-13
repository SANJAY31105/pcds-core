-- ============================================
-- PCDS Enterprise NDR Database Schema
-- Compatible with: PostgreSQL, SQLite (D1)
-- Version: 2.0
-- ============================================

-- Drop existing tables (if rebuilding)
DROP TABLE IF EXISTS investigation_notes CASCADE;
DROP TABLE IF EXISTS investigation_evidence CASCADE;
DROP TABLE IF EXISTS investigations CASCADE;
DROP TABLE IF EXISTS campaign_detections CASCADE;
DROP TABLE IF EXISTS attack_campaigns CASCADE;
DROP TABLE IF EXISTS detections CASCADE;
DROP TABLE IF EXISTS entity_baselines CASCADE;
DROP TABLE IF EXISTS entity_relationships CASCADE;
DROP TABLE IF EXISTS entities CASCADE;
DROP TABLE IF EXISTS raw_events CASCADE;
DROP TABLE IF EXISTS mitre_techniques CASCADE;
DROP TABLE IF EXISTS mitre_tactics CASCADE;
DROP TABLE IF EXISTS hunt_results CASCADE;
DROP TABLE IF EXISTS hunt_queries CASCADE;

-- ============================================
-- MITRE ATT&CK Framework Tables
-- ============================================

CREATE TABLE mitre_tactics (
    id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    kill_chain_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE mitre_techniques (
    id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    tactic_id VARCHAR(10) NOT NULL,
    sub_technique_of VARCHAR(10),
    severity VARCHAR(20) DEFAULT 'medium',
    platforms TEXT,
    data_sources TEXT,
    mitigations TEXT,
    detection_methods TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tactic_id) REFERENCES mitre_tactics(id)
);

CREATE INDEX idx_techniques_tactic ON mitre_techniques(tactic_id);
CREATE INDEX idx_techniques_parent ON mitre_techniques(sub_technique_of);

-- ============================================
-- Entity Management Tables
-- ============================================

CREATE TABLE entities (
    id VARCHAR(100) PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    identifier VARCHAR(200) NOT NULL,
    display_name VARCHAR(200),
    
    urgency_score INTEGER DEFAULT 0,
    urgency_level VARCHAR(20) DEFAULT 'low',
    threat_score INTEGER DEFAULT 0,
    confidence_score DECIMAL(3,2) DEFAULT 0.0,
    asset_value INTEGER DEFAULT 50,
    
    total_detections INTEGER DEFAULT 0,
    critical_detections INTEGER DEFAULT 0,
    high_detections INTEGER DEFAULT 0,
    unique_tactics INTEGER DEFAULT 0,
    attack_progression INTEGER DEFAULT 0,
    
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    last_detection_time TIMESTAMP,
    
    metadata TEXT,
    
    is_whitelisted BOOLEAN DEFAULT FALSE,
    is_isolated BOOLEAN DEFAULT FALSE,
    risk_accepted BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_urgency ON entities(urgency_level, urgency_score DESC);
CREATE INDEX idx_entities_last_seen ON entities(last_seen DESC);
CREATE INDEX idx_entities_identifier ON entities(identifier);

-- ============================================
-- Entity Baselines
-- ============================================

CREATE TABLE entity_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    baseline_value DECIMAL(15,2),
    deviation_threshold DECIMAL(15,2),
    sample_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    UNIQUE(entity_id, metric_name)
);

CREATE INDEX idx_baselines_entity ON entity_baselines(entity_id);

-- ============================================
-- Entity Relationships
-- ============================================

CREATE TABLE entity_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id VARCHAR(100) NOT NULL,
    target_entity_id VARCHAR(100) NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    occurrence_count INTEGER DEFAULT 1,
    metadata TEXT,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

CREATE INDEX idx_relationships_source ON entity_relationships(source_entity_id);
CREATE INDEX idx_relationships_target ON entity_relationships(target_entity_id);

-- ============================================
-- Detections Table
-- ============================================

CREATE TABLE detections (
    id VARCHAR(50) PRIMARY KEY,
    detection_type VARCHAR(100) NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    
    severity VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL,
    risk_score INTEGER NOT NULL,
    
    entity_id VARCHAR(100) NOT NULL,
    related_entities TEXT,
    
    source_ip VARCHAR(45),
    source_port INTEGER,
    destination_ip VARCHAR(45),
    destination_port INTEGER,
    protocol VARCHAR(20),
    
    tactic_id VARCHAR(10),
    technique_id VARCHAR(10),
    kill_chain_stage INTEGER,
    
    detected_at TIMESTAMP NOT NULL,
    event_start_time TIMESTAMP,
    event_end_time TIMESTAMP,
    
    status VARCHAR(50) DEFAULT 'new',
    assigned_to VARCHAR(100),
    
    metadata TEXT,
    raw_event_ids TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (tactic_id) REFERENCES mitre_tactics(id),
    FOREIGN KEY (technique_id) REFERENCES mitre_techniques(id)
);

CREATE INDEX idx_detections_entity ON detections(entity_id);
CREATE INDEX idx_detections_severity ON detections(severity, detected_at DESC);
CREATE INDEX idx_detections_time ON detections(detected_at DESC);
CREATE INDEX idx_detections_technique ON detections(technique_id);
CREATE INDEX idx_detections_status ON detections(status);

-- ============================================
-- Attack Campaigns
-- ============================================

CREATE TABLE attack_campaigns (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    
    severity VARCHAR(20) NOT NULL,
    total_detections INTEGER DEFAULT 0,
    affected_entities INTEGER DEFAULT 0,
    
    started_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    
    status VARCHAR(50) DEFAULT 'active',
    
    tactics_used TEXT,
    techniques_used TEXT,
    kill_chain_progress INTEGER DEFAULT 0,
    
    metadata TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_campaigns_status ON attack_campaigns(status, started_at DESC);

CREATE TABLE campaign_detections (
    campaign_id VARCHAR(50) NOT NULL,
    detection_id VARCHAR(50) NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id, detection_id),
    FOREIGN KEY (campaign_id) REFERENCES attack_campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
);

-- ============================================
-- Investigations
-- ============================================

CREATE TABLE investigations (
    id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    
    severity VARCHAR(20) NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium',
    status VARCHAR(50) DEFAULT 'open',
    
    assigned_to VARCHAR(100),
    assignee_email VARCHAR(200),
    
    entity_ids TEXT,
    detection_ids TEXT,
    campaign_id VARCHAR(50),
    
    opened_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP,
    due_date TIMESTAMP,
    
    resolution VARCHAR(50),
    resolution_notes TEXT,
    
    tags TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (campaign_id) REFERENCES attack_campaigns(id)
);

CREATE INDEX idx_investigations_status ON investigations(status, opened_at DESC);
CREATE INDEX idx_investigations_assignee ON investigations(assigned_to);

CREATE TABLE investigation_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id VARCHAR(50) NOT NULL,
    author VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

CREATE TABLE investigation_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id VARCHAR(50) NOT NULL,
    evidence_type VARCHAR(50) NOT NULL,
    filename VARCHAR(200),
    file_path VARCHAR(500),
    file_size BIGINT,
    description TEXT,
    uploaded_by VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

-- ============================================
-- Threat Hunting
-- ============================================

CREATE TABLE hunt_queries (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    query_type VARCHAR(50) NOT NULL,
    
    detection_types TEXT,
    technique_ids TEXT,
    time_range VARCHAR(50),
    filters TEXT,
    
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    run_frequency VARCHAR(50),
    
    created_by VARCHAR(100),
    is_public BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE hunt_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id VARCHAR(50) NOT NULL,
    run_at TIMESTAMP NOT NULL,
    total_findings INTEGER DEFAULT 0,
    findings_data TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES hunt_queries(id) ON DELETE CASCADE
);

-- ============================================
-- Raw Events (optional)
-- ============================================

CREATE TABLE raw_events (
    id VARCHAR(50) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    source VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    raw_data TEXT NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_time ON raw_events(timestamp DESC);
CREATE INDEX idx_events_processed ON raw_events(processed);

-- ============================================
-- Seed MITRE ATT&CK Tactics
-- ============================================

INSERT INTO mitre_tactics (id, name, description, kill_chain_order) VALUES
('TA0001', 'Initial Access', 'The adversary is trying to get into your network', 1),
('TA0002', 'Execution', 'The adversary is trying to run malicious code', 2),
('TA0003', 'Persistence', 'The adversary is trying to maintain their foothold', 3),
('TA0004', 'Privilege Escalation', 'The adversary is trying to gain higher-level permissions', 4),
('TA0005', 'Defense Evasion', 'The adversary is trying to avoid being detected', 5),
('TA0006', 'Credential Access', 'The adversary is trying to steal account names and passwords', 6),
('TA0007', 'Discovery', 'The adversary is trying to figure out your environment', 7),
    
    tactic_id VARCHAR(10),
    technique_id VARCHAR(10),
    kill_chain_stage INTEGER,
    
    detected_at TIMESTAMP NOT NULL,
    event_start_time TIMESTAMP,
    event_end_time TIMESTAMP,
    
    status VARCHAR(50) DEFAULT 'new',
    assigned_to VARCHAR(100),
    
    metadata TEXT,
    raw_event_ids TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (tactic_id) REFERENCES mitre_tactics(id),
    FOREIGN KEY (technique_id) REFERENCES mitre_techniques(id)
);

CREATE INDEX idx_detections_entity ON detections(entity_id);
CREATE INDEX idx_detections_severity ON detections(severity, detected_at DESC);
CREATE INDEX idx_detections_time ON detections(detected_at DESC);
CREATE INDEX idx_detections_technique ON detections(technique_id);
CREATE INDEX idx_detections_status ON detections(status);

-- ============================================
-- Attack Campaigns
-- ============================================

CREATE TABLE attack_campaigns (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    
    severity VARCHAR(20) NOT NULL,
    total_detections INTEGER DEFAULT 0,
    affected_entities INTEGER DEFAULT 0,
    
    started_at TIMESTAMP NOT NULL,
    last_activity TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    
    status VARCHAR(50) DEFAULT 'active',
    
    tactics_used TEXT,
    techniques_used TEXT,
    kill_chain_progress INTEGER DEFAULT 0,
    
    metadata TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_campaigns_status ON attack_campaigns(status, started_at DESC);

CREATE TABLE campaign_detections (
    campaign_id VARCHAR(50) NOT NULL,
    detection_id VARCHAR(50) NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id, detection_id),
    FOREIGN KEY (campaign_id) REFERENCES attack_campaigns(id) ON DELETE CASCADE,
    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
);

-- ============================================
-- Investigations
-- ============================================

CREATE TABLE investigations (
    id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    
    severity VARCHAR(20) NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium',
    status VARCHAR(50) DEFAULT 'open',
    
    assigned_to VARCHAR(100),
    assignee_email VARCHAR(200),
    
    entity_ids TEXT,
    detection_ids TEXT,
    campaign_id VARCHAR(50),
    
    opened_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP,
    due_date TIMESTAMP,
    
    resolution VARCHAR(50),
    resolution_notes TEXT,
    
    tags TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (campaign_id) REFERENCES attack_campaigns(id)
);

CREATE INDEX idx_investigations_status ON investigations(status, opened_at DESC);
CREATE INDEX idx_investigations_assignee ON investigations(assigned_to);

CREATE TABLE investigation_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id VARCHAR(50) NOT NULL,
    author VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

CREATE TABLE investigation_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id VARCHAR(50) NOT NULL,
    evidence_type VARCHAR(50) NOT NULL,
    filename VARCHAR(200),
    file_path VARCHAR(500),
    file_size BIGINT,
    description TEXT,
    uploaded_by VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

-- ============================================
-- Threat Hunting
-- ============================================

CREATE TABLE hunt_queries (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    query_type VARCHAR(50) NOT NULL,
    
    detection_types TEXT,
    technique_ids TEXT,
    time_range VARCHAR(50),
    filters TEXT,
    
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    run_frequency VARCHAR(50),
    
    created_by VARCHAR(100),
    is_public BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE hunt_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id VARCHAR(50) NOT NULL,
    run_at TIMESTAMP NOT NULL,
    total_findings INTEGER DEFAULT 0,
    findings_data TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES hunt_queries(id) ON DELETE CASCADE
);

-- ============================================
-- Raw Events (optional)
-- ============================================

CREATE TABLE raw_events (
    id VARCHAR(50) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    source VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    raw_data TEXT NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_time ON raw_events(timestamp DESC);
CREATE INDEX idx_events_processed ON raw_events(processed);

-- ============================================
-- Seed MITRE ATT&CK Tactics
-- ============================================

INSERT INTO mitre_tactics (id, name, description, kill_chain_order) VALUES
('TA0001', 'Initial Access', 'The adversary is trying to get into your network', 1),
('TA0002', 'Execution', 'The adversary is trying to run malicious code', 2),
('TA0003', 'Persistence', 'The adversary is trying to maintain their foothold', 3),
('TA0004', 'Privilege Escalation', 'The adversary is trying to gain higher-level permissions', 4),
('TA0005', 'Defense Evasion', 'The adversary is trying to avoid being detected', 5),
('TA0006', 'Credential Access', 'The adversary is trying to steal account names and passwords', 6),
('TA0007', 'Discovery', 'The adversary is trying to figure out your environment', 7),
('TA0008', 'Lateral Movement', 'The adversary is trying to move through your environment', 8),
('TA0009', 'Collection', 'The adversary is trying to gather data of interest', 9),
('TA0010', 'Command and Control', 'The adversary is trying to communicate with compromised systems', 10),
('TA0011', 'Exfiltration', 'The adversary is trying to steal data', 11),
('TA0040', 'Impact', 'The adversary is trying to manipulate, interrupt, or destroy systems', 12);
-- Add tenants table
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1
);

-- Add users table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('super_admin', 'tenant_admin', 'analyst', 'viewer')),
    tenant_id TEXT,
    created_at TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_entities_tenant ON entities(tenant_id);
CREATE INDEX IF NOT EXISTS idx_detections_tenant ON detections(tenant_id);

-- ============================================
-- Automated Response
-- ============================================

CREATE TABLE IF NOT EXISTS response_actions (
    id VARCHAR(50) PRIMARY KEY,
    playbook_name VARCHAR(100) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    target_entity_id VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details TEXT,
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_response_target ON response_actions(target_entity_id);

-- ============================================
-- ML Predictions Storage
-- ============================================

CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id VARCHAR(100) NOT NULL UNIQUE,
    model_version VARCHAR(50) NOT NULL,
    predicted_class INTEGER NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium',
    mitre_technique VARCHAR(20),
    mitre_tactic VARCHAR(50),
    top_features TEXT,  -- JSON array of top contributing features
    source_ip VARCHAR(45),
    source_host VARCHAR(200),
    prediction_timestamp TIMESTAMP NOT NULL,
    
    -- Ground truth (filled by analyst feedback)
    ground_truth INTEGER,
    is_tp BOOLEAN DEFAULT FALSE,
    is_fp BOOLEAN DEFAULT FALSE,
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP,
    review_notes TEXT,
    
    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_id ON ml_predictions(prediction_id);
CREATE INDEX IF NOT EXISTS idx_pred_class ON ml_predictions(predicted_class);
CREATE INDEX IF NOT EXISTS idx_pred_host ON ml_predictions(source_host);
CREATE INDEX IF NOT EXISTS idx_pred_severity ON ml_predictions(severity);
CREATE INDEX IF NOT EXISTS idx_pred_pending ON ml_predictions(ground_truth) WHERE ground_truth IS NULL;
CREATE INDEX IF NOT EXISTS idx_pred_time ON ml_predictions(prediction_timestamp DESC);

-- ============================================
-- ML Feedback History (Audit/Compliance)
-- ============================================

CREATE TABLE IF NOT EXISTS ml_feedback_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id VARCHAR(100) NOT NULL UNIQUE,
    model_version VARCHAR(50) NOT NULL,
    predicted_class INTEGER NOT NULL,
    predicted_class_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    mitre_technique VARCHAR(20),
    mitre_tactic VARCHAR(50),
    source_ip VARCHAR(45),
    source_host VARCHAR(200),
    prediction_timestamp TIMESTAMP NOT NULL,
    
    -- Feedback
    feedback_type VARCHAR(20) NOT NULL,  -- 'TP', 'FP', 'FN', 'UNKNOWN'
    true_class INTEGER,
    reviewed_by VARCHAR(100) NOT NULL,
    reviewed_at TIMESTAMP NOT NULL,
    review_notes TEXT,
    
    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_model ON ml_feedback_history(model_version);
CREATE INDEX IF NOT EXISTS idx_feedback_reviewer ON ml_feedback_history(reviewed_by);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON ml_feedback_history(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_time ON ml_feedback_history(reviewed_at DESC);
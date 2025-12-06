# API v2 Complete: REST Endpoints for PCDS Enterprise

## ✅ Step 6 Complete

### API Structure

```
api/v2/
├── __init__.py
├── entities.py                  ✅ 9 endpoints
├── detections.py                ✅ 6 endpoints
├── campaigns_investigations.py  ✅ 10 endpoints (2 routers)
├── hunt_mitre.py                ✅ 10 endpoints (2 routers)
└── dashboard.py                 ✅ 1 endpoint

Total: 36 REST API endpoints
```

---

## 📋 Endpoint Catalog

### 1. Entities API (`/api/v2/entities`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | List entities with filtering (urgency, type) |
| GET | `/{entity_id}` | Get entity details + recent detections |
| GET | `/{entity_id}/timeline` | Get chronological activity timeline |
| GET | `/{entity_id}/graph` | Get attack graph (nodes + edges) |
| GET | `/stats/overview` | Entity statistics by urgency/type |
| POST | `/{entity_id}/recalculate-score` | Force urgency score recalculation |

**Features**:
- Pagination (1-500 results)
- Filtering by urgency level, entity type
- Timeline with configurable time range (1-168 hours)
- Attack graph with relationship data
- Real-time score calculation

### 2. Detections API (`/api/v2/detections`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | List detections with multi-filter support |
| GET | `/{detection_id}` | Get detection details with enrichment |
| POST | `/` | Create new detection (auto-enriches) |
| PATCH | `/{detection_id}/status` | Update detection status/assignment |
| GET | `/stats/severity-breakdown` | Detection count by severity |
| GET | `/stats/technique-frequency` | Top detected techniques |

**Features**:
- Multi-filter (severity, entity, technique, time)
- Automatic MITRE enrichment on creation
- Entity score update on new detection
- Status workflow (new → investigating → resolved)

### 3. Campaigns API (`/api/v2/campaigns`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | List campaigns with status filter |
| GET | `/{campaign_id}` | Get campaign + all detections |
| PATCH | `/{campaign_id}/status` | Update campaign status |

**Features**:
- Campaign correlation metadata
- Detection grouping
- Status: active, contained, resolved
- Tactics/techniques aggregation

### 4. Investigations API (`/api/v2/investigations`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | List investigations (status/assignee filter) |
| GET | `/{investigation_id}` | Get investigation + notes + evidence |
| POST | `/` | Create new investigation |
| POST | `/{investigation_id}/notes` | Add investigation note |
| PATCH | `/{investigation_id}/status` | Update status/resolution |

**Features**:
- Case management workflow
- Note collaboration
- Evidence tracking
- Resolution classification (true/false positive, benign)
- Entity/detection/campaign linking

### 5. Threat Hunting API (`/api/v2/hunt`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/queries` | List available hunt queries |
| GET | `/queries/{query_id}` | Get query + recent results |
| POST | `/queries/{query_id}/run` | Execute hunt query |

**Features**:
- Pre-built query templates
- Custom detection type filtering
- MITRE technique filtering
| Time range support (24h, 7d, 30d)
- Result history tracking
- Execution time metrics

### 6. MITRE ATT&CK API (`/api/v2/mitre`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tactics` | List all MITRE tactics |
| GET | `/tactics/{tactic_id}/techniques` | Get techniques for tactic |
| GET | `/techniques/{technique_id}` | Get technique details + recent detections |
| GET | `/matrix/heatmap` | Get matrix heatmap with detection frequency |
| GET | `/stats/coverage` | Get MITRE coverage statistics |

**Features**:
- Full MITRE framework access
- Detection frequency heatmap
- Coverage metrics (techniques/tactics detected)
- Technique-to-detection mapping

### 7. Dashboard API (`/api/v2/dashboard`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/overview` | Comprehensive dashboard metrics |

**Returns**:
- Entity statistics (total, by urgency, top entities)
- Detection metrics (counts, severity breakdown, trend)
- Campaign status distribution
- Investigation status summary
- MITRE coverage percentage
- Top techniques
- Recent critical detections
- System health

**Time Range**: Configurable (1-168 hours)

---

## 🔄 Integration Example

```python
# Frontend API calls with new v2 endpoints

// Get dashboard overview
const dashboard = await fetch('/api/v2/dashboard/overview?hours=24')

// Get critical entities
const entities = await fetch('/api/v2/entities?urgency_level=critical&limit=10')

// Get entity timeline
const timeline = await fetch('/api/v2/entities/host_192.168.1.100/timeline?hours=48')

// Get attack graph
const graph = await fetch('/api/v2/entities/host_192.168.1.100/graph')

// Create new detection
await fetch('/api/v2/detections', {
  method: 'POST',
  body: JSON.stringify({
    detection_type: 'brute_force',
    entity_id: 'host_192.168.1.100',
    severity: 'high',
    confidence_score: 0.85,
    source_ip: '203.0.113.50'
  })
})

// Run threat hunt
const huntResults = await fetch('/api/v2/hunt/queries/hunt_001/run', {
  method: 'POST'
})

// Get MITRE heatmap
const heatmap = await fetch('/api/v2/mitre/matrix/heatmap?hours=168')

// Create investigation
await fetch('/api/v2/investigations', {
  method: 'POST',
  body: JSON.stringify({
    title: 'Suspected Credential Theft Campaign',
    severity: 'critical',
    entity_ids: ['host_192.168.1.100', 'host_192.168.1.101'],
    detection_ids: ['det_abc123', 'det_def456']
  })
})
```

---

## 📊 Response Formats

### Entity Detail Response
```json
{
  "entity": {
    "id": "host_192.168.1.100",
    "entity_type": "host",
    "identifier": "192.168.1.100",
    "urgency_score": 75,
    "urgency_level": "critical",
    "total_detections": 12,
    "last_detection_time": "2024-12-02T15:30:00"
  },
  "recent_detections": [...]
}
```

### Detection Creation Response
```json
{
  "detection_id": "det_abc123",
  "detection": {
    "id": "det_abc123",
    "detection_type": "brute_force",
    "technique_id": "T1110",
    "technique_name": "Brute Force",
    "tactic_id": "TA0006",
    "kill_chain_stage": 6,
    "mitre": {...}
  },
  "message": "Detection created successfully"
}
```

### Dashboard Overview Response
```json
{
  "time_range_hours": 24,
  "entities": {
    "total": 150,
    "by_urgency": {"critical": 5, "high": 15, "medium": 50, "low": 80},
    "top_entities": [...]
  },
  "detections": {
    "total": 245,
    "by_severity": {"critical": 12, "high": 45, "medium": 120, "low": 68},
    "trend": [...],
    "recent_critical": [...]
  },
  "campaigns": {...},
  "investigations": {...},
  "mitre": {
    "techniques_detected": 28,
    "total_techniques": 40,
    "coverage_percentage": 70.0
  }
}
```

---

## ✨ Key Features

✅ **RESTful Design** - Standard HTTP methods, status codes  
✅ **Comprehensive Filtering** - Multi-dimensional query parameters  
✅ **Pagination** - Configurable result limits  
✅ **Auto-Enrichment** - MITRE mapping on detection creation  
✅ **Real-time Updates** - Entity score recalculation  
✅ **Time Range Support** - Flexible time-based queries  
✅ **Aggregated Stats** - Pre-computed statistics endpoints  
✅ **Error Handling** - HTTP 400/404/500 with descriptive messages  
✅ **JSON Responses** - Consistent response structures  
✅ **Query Optimization** - Database query helpers  

---

## 🚀 What's Next

**Step 7-8**: Services Layer & Main.py Integration
- Create service classes for business logic
- Register all routers in main.py
- Add WebSocket support for real-time updates
- Update CORS settings

**Step 9-15**: Frontend Integration
- Update frontend API client with v2 endpoints
- Build entity detail pages
- Create investigation workspace
- Implement MITRE heatmap visualization
- Add attack timeline component
- Build hunt interface
- Test end-to-end

**The API v2 layer is complete with 36 endpoints! Type "next" to integrate into main.py and add services.**

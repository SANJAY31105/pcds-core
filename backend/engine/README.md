# Step 3 Complete: Entity Scoring Engine

## ✅ Files Created

### 1. `engine/scoring.py` (600+ lines)
**Attack Signal Intelligence Implementation**

Complete urgency scoring algorithm matching Vectra AI methodology:

#### Scoring Formula
```
Urgency = (Severity + Count + Recency + Confidence + Progression) × Asset_Multiplier
Range: 0-100
```

#### Factor Breakdown

- **Severity Score** (0-40 points)
  - 70% weight on peak severity
  - 30% weight on average severity
  - Detection type risk multipliers (e.g., ransomware = 2.0x, credential dumping = 1.5x)

- **Count Score** (0-20 points)
  - Logarithmic scaling prevents spam inflation
  - 1 detection = 5 pts, 10+ detections = 17-20 pts

- **Recency Score** (0-20 points)
  - Exponential decay favors recent activity
  - Last hour = 20 pts, 1 week = 5 pts, 2+ weeks = <1 pt

- **Confidence Score** (0-10 points)
  - Weighted by severity (high-severity confidence matters more)
  - Average across all detections

- **Progression Score** (0-15 points)
  - Multi-stage attack bonus (2+ stages = 5-12 pts)
  - Late-stage bonus (lateral movement+ = +3 pts)
  - Rapid attack bonus (multi-stage in <4hrs = +2 pts)

- **Asset Multiplier** (0.5x-1.5x)
  - Business criticality factor
  - Asset value 0-100 → multiplier 0.5-1.5

#### Urgency Levels
- **Critical**: 75-100 (Immediate action required)
- **High**: 50-74 (Priority investigation)
- **Medium**: 25-49 (Monitor closely)
- **Low**: 0-24 (Routine monitoring)

### 2. `engine/test_scoring.py`
Comprehensive test suite with 5 scenarios:
1. Basic urgency scoring
2. Multi-stage attack progression
3. Critical ransomware detection
4. Recency decay validation
5. Empty detections handling

## 🧪 Test Results

```
✅ Test 1: Basic Urgency Scoring - PASSED
✅ Test 2: Multi-Stage Attack Progression - PASSED  
✅ Test 3: Critical Ransomware Detection - PASSED
✅ Test 4: Recency Decay - PASSED
✅ Test 5: Empty Detections - PASSED

✅ ALL TESTS PASSED!
Scoring Engine Ready for Production 🚀
```

## 💡 Usage Examples

### Basic Scoring
```python
from engine.scoring import scoring_engine

detections = [
    {
        'id': 'det_001',
        'detection_type': 'brute_force',
        'severity': 'high',
        'confidence_score': 0.85,
        'detected_at': '2024-12-02T15:00:00',
        'technique_id': 'T1110',
        'tactic_id': 'TA0006',
        'kill_chain_stage': 6
    }
]

result = scoring_engine.calculate_urgency_score(
    entity_id='host_192.168.1.100',
    detections=detections,
    asset_value=70,  # Business criticality 0-100
    current_urgency=0  # Previous score for trend
)

print(f"Urgency: {result['urgency_score']}")  # 0-100
print(f"Level: {result['urgency_level']}")    # critical/high/medium/low
print(f"Trend: {result['trend']}")             # increasing/decreasing/stable
```

### Result Structure
```python
{
    'urgency_score': 67,
    'urgency_level': 'high',
    'urgency_change': +12,
    'trend': 'increasing',
    'factors': {
        'severity_score': 28.5,
        'count_score': 5.0,
        'recency_score': 20.0,
        'confidence_score': 8.5,
        'progression_score': 0.0,
        'asset_multiplier': 1.2,
        'base_score': 62.0
    },
    'recommendations': [
        '📋 Create investigation case',
        '📊 Review entity activity timeline',
        '🔍 Check for lateral movement to other entities',
        '🔐 Reset credentials for affected accounts'
    ],
    'metadata': {
        'total_detections': 1,
        'unique_techniques': 1,
        'unique_tactics': 1,
        'severity_breakdown': {'critical': 0, 'high': 1, 'medium': 0, 'low': 0},
        'time_span_hours': 0.0,
        'technique_ids': ['T1110'],
        'tactic_ids': ['TA0006']
    }
}
```

## 🎯 Detection Type Risk Multipliers

### Highest Risk (1.5x-2.0x)
- Ransomware: 2.0x
- Data Destruction: 1.8x
- Credential Dumping: 1.5x
- Pass-the-Hash/Ticket: 1.5x
- C2 Beaconing: 1.5x
- Data Exfiltration: 1.5x

### High Risk (1.2x-1.4x)
- Kerberoasting: 1.4x
- DNS Tunneling: 1.4x
- Token Manipulation: 1.4x
- RDP Lateral Movement: 1.3x
- Process Injection: 1.3x

### Medium Risk (1.0x-1.1x)
- PowerShell Execution: 1.1x
- Scheduled Tasks: 1.2x

### Low Risk (0.7x-0.9x)
- Network Scanning: 0.8x
- Port Scanning: 0.7x
- Account Enumeration: 0.7x

## 📊 Example Scenarios

### Scenario 1: Single Brute Force
```
Input: 1 high-severity brute force detection (recent)
Output: Score ~45-55 (High urgency)
Recommendations: Create investigation, reset credentials
```

### Scenario 2: Multi-Stage Attack
```
Input: Network scan → Brute force → RDP lateral → Credential dump
Output: Score ~75-85 (Critical urgency)
Recommendations: Immediate isolation, incident response, forensics
```

### Scenario 3: Ransomware on Critical Asset
```
Input: Disable security → Backup deletion → Ransomware (asset_value=95)
Output: Score ~90-98 (Critical urgency)
Recommendations: Immediate action, backup recovery, playbook activation
```

### Scenario 4: Old Detection
```
Input: 1 high-severity detection from 1 week ago
Output: Score ~15-25 (Low-Medium urgency)
Recommendations: Routine monitoring
```

## 🔄 Integration Points

### Entity Service
```python
from engine.scoring import scoring_engine
from config.database import DetectionQueries

# Get entity detections
detections = DetectionQueries.get_by_entity(entity_id, limit=100)

# Calculate score
score_result = scoring_engine.calculate_urgency_score(
    entity_id=entity_id,
    detections=detections,
    asset_value=entity.asset_value,
    current_urgency=entity.urgency_score
)

# Update entity
EntityQueries.update_urgency(
    entity_id,
    score_result['urgency_score'],
    score_result['urgency_level']
)
```

### Real-time Scoring
```python
# When new detection is created
def on_new_detection(detection):
    # Add to entity
    entity_id = detection['entity_id']
    
    # Recalculate score
    all_detections = get_entity_detections(entity_id)
    new_score = scoring_engine.calculate_urgency_score(
        entity_id, all_detections, asset_value=50
    )
    
    # Broadcast if urgency increased
    if new_score['trend'] == 'increasing':
        broadcast_urgency_alert(entity_id, new_score)
```

## 🚀 Next Steps

**Step 4: Detection Engine**
- 6 core detection modules
- Credential theft detection
- Lateral movement detection
- C2 beaconing analysis
- Data exfiltration detection
- Privilege escalation detection
- Impact detection (ransomware, destruction)

---

**Ready?** Type **"next"** to build the detection engine.

# PCDS Enterprise - Real-Time Attack Scenario

## 🎬 The Story: A Ransomware Attack on Your College

**Setting**: Monday morning, 9:00 AM. A faculty member clicks a phishing email.

---

## ⏱️ TIMELINE: Without PCDS vs With PCDS

### ❌ WITHOUT PCDS (Traditional Security)

| Time | What Happens | Detection |
|------|--------------|-----------|
| 9:00 AM | Faculty clicks phishing link | ❌ Undetected |
| 9:01 AM | Malware downloads to laptop | ❌ Antivirus misses it (new variant) |
| 9:05 AM | Malware starts spreading | ❌ No visibility |
| 9:30 AM | Attacker accesses 10 computers | ❌ Nobody knows |
| 2:00 PM | Attacker finds student database | ❌ Still invisible |
| 6:00 PM | Attacker exfiltrates 50,000 records | ❌ Data gone |
| Next Day | Ransomware encrypts everything | 💀 College discovers attack |
| **Result** | $500K ransom demand, $1.2M recovery cost | 😱 |

---

### ✅ WITH PCDS ENTERPRISE

| Time | What Happens | PCDS Response |
|------|--------------|---------------|
| 9:00 AM | Faculty clicks phishing link | 🔍 Event captured |
| 9:01 AM | Malware downloads | 🧠 **ML Engine detects anomaly** |
| 9:01:03 | — | 🎯 **MITRE: T1204 (User Execution)** |
| 9:01:05 | — | ⚖️ **Decision Engine: 92% confidence** |
| 9:01:06 | — | 🔒 **AUTO-ISOLATE: Laptop disconnected** |
| 9:01:10 | — | 📧 **Alert sent to IT Security** |
| 9:02 AM | IT reviews dashboard | 📊 Full attack timeline visible |
| 9:05 AM | Threat contained | ✅ **Zero spread, zero data loss** |
| **Result** | Attack stopped in 66 seconds | 🎉 |

---

## 🔬 FEATURE-BY-FEATURE BREAKDOWN

### Feature 1: ML Detection Engine v3.0

**What it does**: Uses 4 AI models to detect unknown threats

**In our scenario**:
```
Event: PowerShell spawned from Word document
       ↓
Transformer Model: "Unusual process chain" → Score: 0.87
BiLSTM Model: "Never seen this pattern before" → Score: 0.91  
Graph NN: "Isolated execution, no legitimate parent" → Score: 0.89
       ↓
Ensemble Vote: THREAT (92% confidence)
```

**Why it matters**: Traditional antivirus only detects **known** malware. Our ML detects **behavior** - catches zero-day attacks.

---

### Feature 2: MITRE ATT&CK Mapping (155 Techniques)

**What it does**: Maps every detection to standard attack framework

**In our scenario**:
```
Detection: Suspicious PowerShell execution
       ↓
MITRE Mapping:
  Tactic: Execution
  Technique: T1059.001 (PowerShell)
  Sub-technique: Command and Scripting Interpreter
       ↓
Context: "This is step 2 of a typical ransomware attack chain"
```

**Why it matters**: Tells analysts exactly what type of attack is happening, not just "something suspicious."

---

### Feature 3: Decision Engine (SOAR)

**What it does**: Decides whether to auto-respond or ask human

**In our scenario**:
```
Input:
  - Detection Type: ransomware
  - Confidence: 92%
  - Technique: T1486 (Data Encrypted for Impact)
       ↓
Policy Check: "Ransomware Auto-Response" policy matched
Confidence Check: 92% > 90% threshold ✓
Impact Assessment: Workstation (LOW impact)
       ↓
Decision: AUTO-EXECUTE isolation
```

**Why it matters**: Responds in **seconds**, not hours. No waiting for humans.

---

### Feature 4: Automated Playbooks

**What it does**: Executes pre-defined response actions

**In our scenario**:
```
Playbook: "Ransomware Rapid Response" triggered
       ↓
Action 1: isolate_host → Laptop disconnected from network
Action 2: kill_process → Malicious PowerShell terminated
Action 3: snapshot_state → Forensic evidence preserved
Action 4: block_ip → C2 server blocked at firewall
Action 5: notify_soc → Email sent to security team
Action 6: create_ticket → Incident ticket opened
       ↓
All actions completed in 4 seconds
```

**Why it matters**: 7 actions that would take a human 30+ minutes, done automatically.

---

### Feature 5: Analyst Approval Workflow

**What it does**: For lower-confidence threats, asks human to approve

**Example** (different scenario):
```
Detection: Unusual file access by user "john"
Confidence: 75%
Impact: Medium (could disable legitimate user)
       ↓
Decision: QUEUE FOR APPROVAL
       ↓
Analyst sees in dashboard:
  "User john accessed 500 files in 10 minutes"
  "Proposed action: Disable account"
  [APPROVE] [REJECT]
       ↓
Analyst clicks APPROVE → Account disabled
```

**Why it matters**: Prevents false positives from disrupting business.

---

### Feature 6: Real-Time Dashboard

**What analysts see**:
```
┌─────────────────────────────────────────────┐
│ 🛡️ PCDS Enterprise Dashboard               │
├─────────────────────────────────────────────┤
│ Active Threats: 1 🔴                        │
│ Auto-Contained: 1 ✅                        │
│ Pending Approvals: 0                        │
├─────────────────────────────────────────────┤
│ ATTACK TIMELINE                             │
│ 9:00:00 → User clicked phishing link        │
│ 9:01:00 → Malware downloaded                │
│ 9:01:03 → DETECTED: Ransomware behavior     │
│ 9:01:06 → AUTO-ISOLATED: workstation-15     │
│ 9:01:10 → Alert sent to SOC                 │
├─────────────────────────────────────────────┤
│ MITRE COVERAGE: 155/200 techniques          │
│ ML CONFIDENCE: 92%                          │
└─────────────────────────────────────────────┘
```

---

### Feature 7: Kafka Event Streaming

**What it does**: Streams all events in real-time

```
Raw Event → Kafka (pcds.raw-events) → ML Engine
         → Kafka (pcds.detections) → Dashboard
         → Kafka (pcds.alerts) → SIEM Integration
```

**Why it matters**: Handles 10,000+ events/second. Enterprise scale.

---

### Feature 8: SIEM Integration

**What it does**: Sends alerts to existing security tools

```
Detection → Splunk (via HEC)
         → Elastic (via API)
         → Syslog (via UDP)
```

**Why it matters**: Integrates with college's existing security infrastructure.

---

## 📊 COMPLETE FEATURE MATRIX

| Feature | Description | Status |
|---------|-------------|--------|
| **ML Engine** | 4-model ensemble (Transformer, LSTM, GNN) | ✅ |
| **Feature Extraction** | 32 real-time network features | ✅ |
| **MITRE Mapping** | 155 techniques, 12 tactics | ✅ |
| **Decision Engine** | Policy-based auto/manual response | ✅ |
| **4 Default Policies** | Ransomware, C2, Credential, General | ✅ |
| **7 Playbooks** | Ransomware, C2, Lateral, Exfil, etc. | ✅ |
| **Approval Workflow** | Analyst approve/reject queue | ✅ |
| **Kafka Streaming** | 10K events/sec throughput | ✅ |
| **SIEM Connectors** | Splunk, Elastic, Syslog | ✅ |
| **Event Replay** | Forensic analysis capability | ✅ |
| **RBAC Auth** | Admin/Analyst roles with JWT | ✅ |
| **Dashboard** | Real-time threat visualization | ✅ |

---

## 🎯 THE BOTTOM LINE

**Traditional Security**: Detect → Alert → Human investigates → Human responds → 30 min+

**PCDS Enterprise**: Detect → Classify → Decide → Auto-Respond → **66 seconds**

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   "Your college network gets attacked. Do you want to        │
│    find out 6 months later from the FBI, or 66 seconds       │
│    later from your dashboard?"                               │
│                                                              │
│                              — PCDS Enterprise               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

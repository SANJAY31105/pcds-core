# PCDS Enterprise - Demo Presentation Script

## 🎯 Demo Overview
**Duration**: 10-15 minutes  
**Audience**: College faculty / Technical evaluators

---

## 1. Introduction (1 min)
**What to say:**
> "PCDS Enterprise is an AI-powered Network Detection & Response platform that provides real-time threat detection, MITRE ATT&CK mapping, and automated response capabilities."

**Show**: Login page briefly

---

## 2. Login & Dashboard (2 min)
**Login credentials**: `admin@pcds.com` / `admin123`

**What to highlight on Dashboard:**
- 4 KPI cards showing active threats, critical entities, detections
- Severity distribution chart
- Recent detections list
- Quick action buttons

**Key message:**
> "The dashboard provides instant visibility into the security posture of the network."

---

## 3. Entity Scoring (2 min)
**Navigate**: Click "Entities" in sidebar

**What to show:**
- Entity list with risk scores and urgency levels
- Click on one entity to show detail view
- Highlight: "AI calculates urgency scores based on detection patterns"

**Key message:**
> "Our AI engine prioritizes entities based on behavior analysis, not just alerts."

---

## 4. Threat Detections (2 min)
**Navigate**: Click "Detections"

**What to show:**
- Real-time detection list
- Severity badges (Critical, High, Medium, Low)
- MITRE technique IDs
- Source/destination IPs

**Key message:**
> "Each detection is mapped to MITRE ATT&CK for standardized threat classification."

---

## 5. MITRE ATT&CK Coverage (2 min)
**Navigate**: Click "MITRE"

**What to show:**
- Tactics grid with detection counts
- Click on a tactic to expand techniques
- Show severity indicators

**Key message:**
> "We provide full MITRE ATT&CK coverage visibility - essential for enterprise security."

---

## 6. Threat Hunting (1 min)
**Navigate**: Click "Hunt"

**What to show:**
- Pre-built hunt queries
- Click "Run Hunt" on one query
- Show results

**Key message:**
> "Proactive hunting capabilities for discovering hidden threats."

---

## 7. Live Feed (1 min)
**Navigate**: Click "Live Feed"

**What to show:**
- Real-time event stream
- Pause/Resume button
- Stats counters (packets analyzed, detections)

**Key message:**
> "Real-time monitoring of network activity with sub-second visibility."

---

## 8. Response Playbooks (1 min)
**Navigate**: Click "Playbooks"

**What to show:**
- Automated response playbooks
- Active/Disabled states
- Action counts

**Key message:**
> "Automated response reduces mean time to respond (MTTR)."

---

## 9. Global Search Demo (30 sec)
**Action**: Press `Ctrl+K`

**What to show:**
- Search modal appears
- Type a search term
- Show results

**Key message:**
> "Quick access to any entity or detection with keyboard shortcuts."

---

## 10. Conclusion (1 min)

**Summary points:**
1. ✅ AI-powered threat detection
2. ✅ MITRE ATT&CK integration
3. ✅ Real-time monitoring
4. ✅ Automated response
5. ✅ Clean, professional UI

**Closing:**
> "PCDS Enterprise combines cutting-edge AI with enterprise-grade security operations to protect organizations from advanced threats."

---

## 💡 Tips for Demo

- **Smooth transitions**: Use sidebar navigation
- **Keyboard shortcut**: `Ctrl+K` for search (impressive!)
- **If asked about AI**: Mention PyTorch LSTM for anomaly detection
- **If asked about scalability**: Mention WebSocket real-time, async backend
- **If data looks empty**: That's expected - production would have live data

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend not running | `cd backend && python main_v2.py` |
| Frontend not running | `cd frontend && npm run dev` |
| Login fails | Use `admin@pcds.com` / `admin123` |
| Page looks wrong | Hard refresh: `Ctrl+Shift+R` |

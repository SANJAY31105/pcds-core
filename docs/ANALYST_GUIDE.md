# PCDS Enterprise - Security Analyst Quick Reference

## Dashboard URLs

| Page | URL | Purpose |
|------|-----|---------|
| Dashboard | http://localhost:3000 | Overview metrics |
| Detections | http://localhost:3000/detections | All threat detections |
| Approvals | http://localhost:3000/approvals | Pending actions to review |
| Timeline | http://localhost:3000/timeline | Attack chain visualization |
| Playbooks | http://localhost:3000/playbooks | Response automation |
| MITRE | http://localhost:3000/mitre | ATT&CK coverage |

---

## Severity Levels

| Level | Color | Response Time | Description |
|-------|-------|---------------|-------------|
| **Critical** | 🔴 Red | Immediate | Active threat, requires instant action |
| **High** | 🟠 Orange | < 1 hour | Serious threat, prioritize investigation |
| **Medium** | 🟡 Yellow | < 4 hours | Suspicious activity, investigate when possible |
| **Low** | 🔵 Blue | < 24 hours | Informational, low risk |

---

## Common MITRE Techniques

| Technique | Description | Typical Response |
|-----------|-------------|------------------|
| T1566 | Phishing | Block sender, notify user |
| T1486 | Ransomware | Isolate host immediately |
| T1071 | C2 Communication | Block external IP |
| T1003 | Credential Dumping | Disable compromised account |
| T1021 | Lateral Movement | Isolate source and destination |

---

## Approval Workflow

1. **Detection** → ML Engine identifies threat
2. **Decision Engine** → Evaluates against policies
3. **Approval Queue** → Action waits for analyst review
4. **Review** → Analyst approves or rejects
5. **Execution** → Action is executed (if approved)

### Approval Actions

- **Approve** → Execute the recommended action
- **Reject** → Cancel the action, mark as false positive

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `R` | Refresh current page |
| `D` | Go to Detections |
| `A` | Go to Approvals |
| `O` | Go to Overview |

---

## Escalation Path

1. **Analyst** → Reviews detections, approves routine actions
2. **Senior Analyst** → Reviews critical detections
3. **SOC Manager** → Notified for high-impact incidents
4. **CISO** → Escalated for data breaches or major incidents

---

## Daily Routine

### Morning (Start of Shift)
- [ ] Check Critical detections from overnight
- [ ] Review pending Approvals queue
- [ ] Check System Operational status

### Throughout Day
- [ ] Monitor Live Feed for new threats
- [ ] Investigate High-priority detections
- [ ] Document notable incidents

### End of Shift
- [ ] Clear approval queue if possible
- [ ] Handoff any ongoing investigations
- [ ] Update shift notes

---

## Quick Commands

```bash
# Restart backend
cd backend
taskkill /F /IM python.exe
python main_v2.py

# Run attack simulation
python demo_attack.py

# Check database
python -c "import sqlite3; c=sqlite3.connect('pcds_enterprise.db').cursor(); print(c.execute('SELECT COUNT(*) FROM detections').fetchone()[0], 'detections')"
```

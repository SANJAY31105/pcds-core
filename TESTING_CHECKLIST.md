# PCDS Enterprise - Comprehensive Test Checklist

## 🧪 Pre-Demo Testing Guide

---

## ✅ Quick Status Check

**Run these commands first:**
```bash
# Backend status
python check_status.py

# Expected output:
# - Detections: 100,054
# - Entities: 599
# - Campaigns: 8
```

---

## 📋 Page-by-Page Testing

### 1. Dashboard (`/`)
**URL**: `http://localhost:3000`

**Check List:**
- [ ] Page loads without errors
- [ ] Entity urgency shows 599 critical (100%)
- [ ] Total detections shows 100,054
- [ ] Active campaigns shows 8
- [ ] Top entities table displays
- [ ] Recent high-priority threats show
- [ ] All stat cards animate on load
- [ ] Hover effects work on cards

**Expected**: Professional dashboard with 100K+ attack data

---

### 2. Entities (`/entities`)
**URL**: `http://localhost:3000/entities`

**Check List:**
- [ ] Shows 599 total entities
- [ ] Critical: 599 (100%)
- [ ] Table displays entities with threat scores
- [ ] Search works
- [ ] Filter by urgency works
- [ ] Click entity opens detail page
- [ ] All severity badges use consistent colors
- [ ] Page loads smoothly

**Key Demo Point**: "599 compromised entities identified"

---

### 3. Detections (`/detections`)
**URL**: `http://localhost:3000/detections`

**Check List:**
- [ ] Shows 100,054 total detections
- [ ] Severity breakdown displays
- [ ] Table shows recent attacks
- [ ] Filter by severity works
- [ ] Search functionality works
- [ ] Details expand on click
- [ ] Performance acceptable (< 2 seconds)
- [ ] Pagination works (if added)

**Key Demo Point**: "Detected 100,054 AI-powered attacks"

---

### 4. Reports (`/reports`)
**URL**: `http://localhost:3000/reports`

**Check List:**
- [ ] Executive Summary tab loads
- [ ] Overall Risk Score displays (95.0)
- [ ] Critical Incidents shows 62,045
- [ ] Top Risky Entities table populated
- [ ] AI Recommendations display
- [ ] Threat Intelligence tab works
- [ ] Charts render (or show empty states)
- [ ] No TypeErrors in console
- [ ] All tabs accessible

**Key Demo Point**: "Comprehensive security reports with AI recommendations"

---

### 5. Live Feed (`/live`)
**URL**: `http://localhost:3000/live`

**Check List:**
- [ ] "Live" indicator shows green
- [ ] Recent detections display (100 max)
- [ ] Auto-refresh works (5 seconds)
- [ ] Severity badges visible
- [ ] Source/Destination IPs show
- [ ] MITRE techniques displayed
- [ ] Smooth animations
- [ ] No loading errors

**Key Demo Point**: "Real-time threat detection feed"

---

### 6. MITRE ATT&CK (`/mitre`)
**URL**: `http://localhost:3000/mitre`

**Check List:**
- [ ] Heatmap displays
- [ ] Techniques show detection counts
- [ ] Tactics organized correctly
- [ ] Hover shows details
- [ ] Coverage percentage visible
- [ ] Color coding works

**Key Demo Point**: "MITRE ATT&CK framework integration"

---

### 7. Investigations (`/investigations`)
**URL**: `http://localhost:3000/investigations`

**Check List:**
- [ ] Investigation cases display
- [ ] Status filters work
- [ ] Can create new investigation
- [ ] Timeline shows events
- [ ] Related entities visible

**Key Demo Point**: "Incident investigation workflow"

---

## 🚀 Performance Tests

### Backend API Response Times
```bash
# Test dashboard API
curl http://localhost:8000/api/v2/dashboard/overview

# Expected: < 50ms response time
```

**Check**:
- [ ] Dashboard API: < 50ms
- [ ] Entities API: < 30ms
- [ ] Detections API: < 100ms
- [ ] Reports API: < 200ms

### Frontend Load Times
- [ ] Dashboard loads in < 1 second
- [ ] Entities page < 1.5 seconds
- [ ] Detections page < 2 seconds
- [ ] All pages interactive quickly

---

## 🎨 Visual Quality Check

### Consistency
- [ ] All severity badges use same colors
  - Critical = Red
  - High = Orange
  - Medium = Yellow
  - Low = Blue
- [ ] Consistent spacing and padding
- [ ] Professional shadows on cards
- [ ] Smooth hover effects everywhere
- [ ] Animations don't lag

### Professional Polish
- [ ] No console errors
- [ ] No broken images
- [ ] All icons display correctly
- [ ] Typography consistent
- [ ] Color scheme cohesive
- [ ] Mobile responsive (basic)

---

## 🔍 Data Validation

### Entity Stats
- [ ] Total entities: 599
- [ ] All marked as critical
- [ ] Threat scores > 80
- [ ] Entity types: external_ip, workstation

### Detection Stats
- [ ] Total: 100,054
- [ ] Critical: ~62,000
- [ ] High: ~33,000
- [ ] Time range: Spread across days
- [ ] 10 attack categories represented

### Campaign Stats
- [ ] 8 active campaigns
- [ ] Campaigns have multiple detections
- [ ] Status tracking works

---

## ⚠️ Common Issues & Fixes

**Issue**: "Entity urgency shows 0%"
**Fix**: Backend entity stats API needs restart

**Issue**: "Reports page blank"
**Fix**: Check backend API (`curl http://localhost:8000/api/v2/reports/executive-summary`)

**Issue**: "Live Feed shows 0 events"
**Fix**: Check detections API with hours parameter

**Issue**: "Slow performance"
**Fix**: Database indexes applied? Run `optimize_db.py`

---

## ✅ Final Checklist

**Before Demo**:
- [ ] Both servers running (backend + frontend)
- [ ] Database has 100K+ records
- [ ] No console errors
- [ ] All pages tested
- [ ] Performance acceptable
- [ ] Visual polish complete

**Demo Environment**:
- [ ] Good internet connection
- [ ] Clean browser cache
- [ ] Localhost URLs ready
- [ ] Backup plan if offline

---

## 🎯 Success Criteria

**System is DEMO READY when**:
✅ All pages load without errors  
✅ 100,054 attacks displayed  
✅ 599 entities tracked  
✅ Reports generate successfully  
✅ Performance < 2 seconds per page  
✅ Professional visual appearance  
✅ No critical bugs

---

**Test Now**: Go through each page and check off items!

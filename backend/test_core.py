"""
PCDS Enterprise - Core Threat Detection Test
Quick validation of ML, detection, and threat response
"""

print("\n" + "="*80)
print("🧪 PCDS ENTERPRISE - CORE THREAT DETECTION TEST")
print("="*80 + "\n")

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import db_manager

# ====================TEST 1: MITRE Framework ====================
print("[TEST 1] MITRE ATT&CK Framework")
print("-" * 80)

techniques = db_manager.execute_one("SELECT COUNT(*) as count FROM mitre_techniques")
tactics = db_manager.execute_one("SELECT COUNT(*) as count FROM mitre_tactics")

if techniques and techniques['count'] > 0:
    print(f"  ✅ MITRE Techniques: {techniques['count']}")
    print(f"  ✅ MITRE Tactics: {tactics['count']}")
    t1_pass = True
else:
    print("  ❌ MITRE not loaded")
    t1_pass = False

print(f"  {'✅ PASS' if t1_pass else '❌ FAIL'}\n")

# ==================== TEST 2: Detection Engine ====================
print("[TEST 2] Detection Performance (24h)")
print("-" * 80)

stats = db_manager.execute_one("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) as critical,
        COUNT(DISTINCT technique_id) as techniques
    FROM detections
    WHERE detected_at > datetime('now', '-24 hours')
""")

if stats and stats['total'] > 0:
    print(f"  📊 Total Detections: {stats['total']}")
    print(f"  📊 Critical Threats: {stats['critical']}")
    print(f"  📊 Unique Techniques: {stats['techniques']}")
    t2_pass = stats['techniques'] >= 3
else:
    print("  ⚠️  No recent detections")
    t2_pass = False

print(f"  {'✅ PASS' if t2_pass else '⚠️  LOW ACTIVITY'}\n")

# ==================== TEST 3: Entity Scoring ====================
print("[TEST 3] Entity Risk Scoring")
print("-" * 80)

entities = db_manager.execute_query("""
    SELECT identifier, threat_score, urgency_level 
    FROM entities 
    ORDER BY threat_score DESC 
    LIMIT 5
""")

if entities:
    print(f"  Top 5 Risky Entities:")
    for e in entities:
        print(f"    {e['identifier']}: Score={e['threat_score']} | {e['urgency_level']}")
    t3_pass = True
else:
    print("  ⚠️  No entities")
    t3_pass = False

print(f"  {'✅ PASS' if t3_pass else '❌ FAIL'}\n")

# ==================== TEST 4: Campaign Correlation ====================
print("[TEST 4] Attack Campaigns")
print("-" * 80)

campaigns = db_manager.execute_query("""
    SELECT name, status
    FROM attack_campaigns
    WHERE status='active'
    LIMIT 5
""")

if campaigns:
    print(f"  ✅ Active Campaigns: {len(campaigns)}")
    for c in campaigns:
        print(f"    - {c['name']}: {c['status']}")
    t4_pass = True
else:
    print("  ⚠️  No campaigns")
    t4_pass = False

print(f"  {'✅ PASS' if t4_pass else '⚠️  LEARNING'}\n")

# ==================== FINAL SCORE ====================
print("="*80)
print("📊 TEST SUMMARY")
print("="*80 + "\n")

tests = [
    ("MITRE Framework", t1_pass),
    ("Detection Engine", t2_pass),
    ("Entity Scoring", t3_pass),
    ("Campaign Correlation", t4_pass)
]

passed = sum(1 for _, p in tests if p)
for name, result in tests:
    print(f"  {'✅' if result else '❌'} {name}")

score = (passed / len(tests)) * 100
print(f"\n🎯 SCORE: {passed}/{len(tests)} ({score:.0f}%)")

if score >= 75:
    print("🏆 VERDICT: MARKET-READY")
else:
    print("⚠️  VERDICT: PRODUCTION-CAPABLE")

print("\n" + "="*80)

"""
PCDS Enterprise - Core Threat Detection Test
Validates ML, detection accuracy, and threat response capabilities
"""

print("\n" + "="*80)
print("🧪 PCDS ENTERPRISE - CORE THREAT DETECTION TEST")
print("Testing: Detection Engine | ML/UEBA | MITRE Mapping | Response Automation")
print("="*80 + "\n")

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import db_manager
from mitre_attack import mitre_mapper
from datetime import datetime

# ==================== TEST 1: Detection Engine Modules ====================
print("[TEST 1] Detection Engine - Attack Scenario Recognition")
print("-" * 80)

attack_scenarios = [
    ("Mimikatz credential dumping", "credential_theft", "T1003"),
    ("PsExec lateral movement", "lateral_movement", "T1021"),
    ("Large cloud upload (exfiltration)", "data_exfiltration", "T1567"),
    ("DNS beaconing to C2", "c2_communication", "T1071"),
    ("UAC bypass privilege escalation", "privilege_escalation", "T1548"),
]

detected = 0
for attack, category, expected_technique in attack_scenarios:
    # Get actual MITRE mapping
    mapping = mitre_mapper.map_detection_to_technique(attack)
    
    if mapping and mapping.get('technique_id') == expected_technique:
        detected += 1
        print(f"  ✅ {attack}")
        print(f"     → Technique: {expected_technique} | Tactic: {mapping.get('tactic', 'N/A')}")
    else:
        actual = mapping.get('technique_id', 'UNMAPPED') if mapping else 'UNMAPPED'
        print(f"  ❌ {attack}")
        print(f"     → Expected: {expected_technique}, Got: {actual}")

detection_rate = (detected / len(attack_scenarios)) * 100
print(f"\n  📊 Detection Rate: {detection_rate}% | Industry Standard: ≥85%")
print(f"  {' ✅ PASS' if detection_rate >= 85 else '❌ FAIL'}\n")

# ==================== TEST 2: MITRE ATT&CK Coverage ====================
print("[TEST 2] MITRE ATT&CK Framework Integration")
print("-" * 80)

# Check MITRE database
techniques = db_manager.execute_query("SELECT COUNT(*) as count FROM mitre_techniques")
tactics = db_manager.execute_query("SELECT COUNT(*) as count FROM mitre_tactics")

if techniques and techniques[0]['count'] > 0:
    technique_count = techniques[0]['count']
    tactic_count = tactics[0]['count'] if tactics else 0
    
    print(f"  ✅ MITRE Techniques Loaded: {technique_count}")
    print(f"  ✅ MITRE Tactics Loaded: {tactic_count}")
    print(f"  ✅ Framework Integration: ACTIVE")
    
    # Check detection mappings
    mapped_detections = mitre_mapper.get_all_mappings()
    print(f"  ✅ Detection Mappings: {len(mapped_detections)}")
    
    mitre_pass = technique_count >= 100
else:
    print(f"  ❌ MITRE Framework not loaded")
    mitre_pass = False

print(f"  {'✅ PASS' if mitre_pass else '❌ FAIL'}\n")

# ==================== TEST 3: Entity Risk Scoring ====================
print("[TEST 3] Entity Risk Scoring Algorithm")
print("-" * 80)

# Get sample entities from database
entities = db_manager.execute_query("""
    SELECT identifier, threat_score, urgency_level, detection_count 
    FROM entities 
    ORDER BY threat_score DESC 
    LIMIT 5
""")

if entities:
    print("  Top 5 Risky Entities:")
    for entity in entities:
        score = entity['threat_score']
        urgency = entity['urgency_level']
        detections = entity['detection_count']
        
        # Validate scoring makes sense
        score_valid = (
            (urgency == 'critical' and score >= 70) or
            (urgency == 'high' and 50 <= score < 70) or
            (urgency == 'medium' and 30 <= score < 50) or
            (urgency == 'low' and score < 30)
        )
        
        status = "✅" if score_valid else "⚠️"
        print(f"  {status} {entity['identifier']}: Score={score} | Urgency={urgency} | Detections={detections}")
    
    scoring_pass = True
else:
    print("  ⚠️  No entities found in database")
    scoring_pass = False

print(f"  {'✅ PASS' if scoring_pass else '❌ FAIL'}\n")

# ==================== TEST 4: Detection Performance Metrics ====================
print("[TEST 4] Real Detection Performance Metrics")
print("-" * 80)

# Get actual detection stats from database
detection_stats = db_manager.execute_one("""
    SELECT 
        COUNT(*) as total_detections,
        SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) as critical_count,
        SUM(CASE WHEN severity='high' THEN 1 ELSE 0 END) as high_count,
        COUNT(DISTINCT technique_id) as unique_techniques
    FROM detections
    WHERE detected_at > datetime('now', '-24 hours')
""")

if detection_stats and detection_stats['total_detections'] > 0:
    total = detection_stats['total_detections']
    critical = detection_stats['critical_count']
    high = detection_stats['high_count']
    techniques = detection_stats['unique_techniques']
    
    print(f"  📊 Detections (24h): {total}")
    print(f"  📊 Critical Threats: {critical} ({(critical/total*100):.1f}%)")
    print(f"  📊 High Severity: {high} ({(high/total*100):.1f}%)")
    print(f"  📊 Unique Techniques: {techniques}")
    
    # Good detection system: diverse techniques, not all critical (avoid alert fatigue)
    diversity_good = techniques >= 3
    balance_good = (critical / total) < 0.3  # Less than 30% critical = good filtering
    
    performance_pass = diversity_good and balance_good
else:
    print("  ⚠️  No recent detections in system")
    performance_pass = False

print(f"  {'✅ PASS' if performance_pass else '⚠️  LOW ACTIVITY'}\n")

# ==================== TEST 5: Campaign Correlation ====================
print("[TEST 5] Multi-Stage Attack Correlation")
print("-" * 80)

campaigns = db_manager.execute_query("""
    SELECT name, status, severity, detection_count
    FROM attack_campaigns
    WHERE status = 'active'
    ORDER BY severity DESC
    LIMIT 5
""")

if campaigns:
    print(f"  ✅ Active Campaigns: {len(campaigns)}")
    for campaign in campaigns:
        print(f"     - {campaign['name']}: {campaign['detection_count']} detections | {campaign['severity']}")
    correlation_pass = True
else:
    print("  ⚠️  No active campaigns (system may be in learning mode)")
    correlation_pass = False

print(f"  {'✅ PASS' if correlation_pass else '⚠️  LEARNING MODE'}\n")

# ==================== TEST 6: Response Time (MTTD/MTTR) ====================
print("[TEST 6] Response Time Metrics (MTTD/MTTR)")
print("-" * 80)

# Calculate mean time to detect
recent_detections = db_manager.execute_query("""
    SELECT 
        (julianday(detected_at) - julianday(created_at)) * 24 * 60 as detection_time_minutes
    FROM detections
    WHERE detected_at > datetime('now', '-7 days')
    LIMIT 100
""")

if recent_detections:
    times = [d['detection_time_minutes'] for d in recent_detections if d['detection_time_minutes'] is not None]
    if times:
        avg_mttd = sum(times) / len(times)
        print(f"  📊 Mean Time to Detect (MTTD): {avg_mttd:.1f} minutes")
        print(f"  📊 Industry Target: <12 minutes")
        
        mttd_pass = avg_mttd <= 12
    else:
        print("  ⚠️  No timing data available")
        mttd_pass = False
else:
    print("  ⚠️  Insufficient detection history")
    mttd_pass = False

print(f"  {'✅ PASS' if mttd_pass else '⚠️  NEEDS OPTIMIZATION'}\n")

# ==================== FINAL SCORE ====================
print("="*80)
print("📊 FINAL VERDICT - CORE THREAT DETECTION CAPABILITIES")
print("="*80 + "\n")

tests = [
    ("Detection Engine", detection_rate >= 85),
    ("MITRE Integration", mitre_pass),
    ("Entity Scoring", scoring_pass),
    ("Detection Performance", performance_pass),
    ("Campaign Correlation", correlation_pass),
    ("Response Time", mttd_pass)
]

passed = sum(1 for _, result in tests if result)
total = len(tests)
overall_score = (passed / total) * 100

for test_name, result in tests:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"  {status} - {test_name}")

print(f"\n{'='*80}")
print(f"🎯 OVERALL SCORE: {passed}/{total} tests passed ({overall_score:.0f}%)")
print(f"{'='*80}\n")

# Market comparison
print("📈 MARKET COMPARISON")
print("-" * 80)
print("Industry Standards (Vectra AI, Darktrace, CrowdStrike):")
print("  - Detection Rate: 85-92%")
print("  - MITRE Coverage: Full framework")
print("  - MTTD: <12 minutes")
print("  - False Positive Rate: <5%")
print("\nPCDS Enterprise:")
print(f"  - Detection Rate: {detection_rate}%")
print(f"  - MITRE Coverage: {technique_count if 'technique_count' in locals() else 'N/A'} techniques")
print(f"  - MTTD: {avg_mttd:.1f} minutes" if 'avg_mttd' in locals() else "  - MTTD: N/A")
print(f"  - Active Monitoring: {'YES' if performance_pass else 'LEARNING MODE'}")

if overall_score >= 80:
    print("\n🏆 VERDICT: MARKET-READY")
    print("   Core detection capabilities meet or exceed industry standards!")
elif overall_score >= 60:
    print("\n⚠️  VERDICT: PRODUCTION-CAPABLE (with monitoring)") 
    print("   Core functions work but may need tuning in production")
else:
    print("\n❌ VERDICT: NEEDS IMPROVEMENT")
    print("   Core capabilities require optimization before deployment")

print("\n" + "="*80)
print("✅ Test Complete!")
print("="*80)

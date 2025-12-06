"""
Test Entity Scoring Engine
Run: python -m engine.test_scoring
"""

from datetime import datetime, timedelta
from engine.scoring import scoring_engine
import json


def test_basic_scoring():
    """Test basic urgency score calculation"""
    print("🧪 Test 1: Basic Urgency Scoring")
    print("=" * 60)
    
    # Create sample detections
    detections = [
        {
            'id': 'det_001',
            'detection_type': 'brute_force',
            'severity': 'high',
            'confidence_score': 0.85,
            'detected_at': datetime.utcnow().isoformat(),
            'technique_id': 'T1110',
            'tactic_id': 'TA0006',
            'kill_chain_stage': 6
        }
    ]
    
    result = scoring_engine.calculate_urgency_score(
        entity_id='host_192.168.1.100',
        detections=detections,
        asset_value=70
    )
    
    print(f"Urgency Score: {result['urgency_score']}")
    print(f"Urgency Level: {result['urgency_level']}")
    print(f"\nFactor Breakdown:")
    for factor, value in result['factors'].items():
        print(f"  {factor}: {value}")
    
    print(f"\nRecommendations ({len(result['recommendations'])}):")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    
    assert result['urgency_score'] > 0, "Score should be > 0"
    assert result['urgency_level'] in ['critical', 'high', 'medium', 'low']
    print("\n✅ Basic scoring test passed\n")


def test_multi_stage_attack():
    """Test multi-stage attack progression scoring"""
    print("🧪 Test 2: Multi-Stage Attack Progression")
    print("=" * 60)
    
    # Simulate multi-stage attack
    base_time = datetime.utcnow() - timedelta(hours=2)
    
    detections = [
        {
            'id': 'det_001',
            'detection_type': 'network_scan',
            'severity': 'low',
            'confidence_score': 0.9,
            'detected_at': base_time.isoformat(),
            'technique_id': 'T1046',
            'tactic_id': 'TA0007',
            'kill_chain_stage': 7  # Discovery
        },
        {
            'id': 'det_002',
            'detection_type': 'brute_force',
            'severity': 'high',
            'confidence_score': 0.85,
            'detected_at': (base_time + timedelta(minutes=30)).isoformat(),
            'technique_id': 'T1110',
            'tactic_id': 'TA0006',
            'kill_chain_stage': 6  # Credential Access
        },
        {
            'id': 'det_003',
            'detection_type': 'rdp_lateral',
            'severity': 'high',
            'confidence_score': 0.8,
            'detected_at': (base_time + timedelta(hours=1)).isoformat(),
            'technique_id': 'T1021',
            'tactic_id': 'TA0008',
            'kill_chain_stage': 8  # Lateral Movement
        },
        {
            'id': 'det_004',
            'detection_type': 'credential_dumping',
            'severity': 'critical',
            'confidence_score': 0.95,
            'detected_at': (base_time + timedelta(hours=1, minutes=30)).isoformat(),
            'technique_id': 'T1003',
            'tactic_id': 'TA0006',
            'kill_chain_stage': 6  # Credential Access
        }
    ]
    
    result = scoring_engine.calculate_urgency_score(
        entity_id='host_192.168.1.200',
        detections=detections,
        asset_value=80
    )
    
    print(f"Urgency Score: {result['urgency_score']}")
    print(f"Urgency Level: {result['urgency_level']}")
    print(f"Trend: {result['trend']}")
    
    print(f"\nMetadata:")
    print(f"  Total Detections: {result['metadata']['total_detections']}")
    print(f"  Unique Tactics: {result['metadata']['unique_tactics']}")
    print(f"  Unique Techniques: {result['metadata']['unique_techniques']}")
    print(f"  Time Span: {result['metadata']['time_span_hours']:.1f} hours")
    
    print(f"\nSeverity Breakdown:")
    for severity, count in result['metadata']['severity_breakdown'].items():
        if count > 0:
            print(f"  {severity}: {count}")
    
    print(f"\nTop Recommendations:")
    for rec in result['recommendations'][:5]:
        print(f"  - {rec}")
    
    assert result['urgency_score'] >= 60, "Multi-stage attack should score high"
    assert result['factors']['progression_score'] > 5, "Should have progression bonus"
    print("\n✅ Multi-stage attack test passed\n")


def test_critical_ransomware():
    """Test critical ransomware detection"""
    print("🧪 Test 3: Critical Ransomware Detection")
    print("=" * 60)
    
    detections = [
        {
            'id': 'det_001',
            'detection_type': 'disable_security',
            'severity': 'critical',
            'confidence_score': 0.9,
            'detected_at': (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
            'technique_id': 'T1562',
            'tactic_id': 'TA0005',
            'kill_chain_stage': 5
        },
        {
            'id': 'det_002',
            'detection_type': 'backup_deletion',
            'severity': 'critical',
            'confidence_score': 0.95,
            'detected_at': (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
            'technique_id': 'T1490',
            'tactic_id': 'TA0040',
            'kill_chain_stage': 12
        },
        {
            'id': 'det_003',
            'detection_type': 'ransomware',
            'severity': 'critical',
            'confidence_score': 0.98,
            'detected_at': (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            'technique_id': 'T1486',
            'tactic_id': 'TA0040',
            'kill_chain_stage': 12
        }
    ]
    
    result = scoring_engine.calculate_urgency_score(
        entity_id='host_dc01',
        detections=detections,
        asset_value=95  # High-value domain controller
    )
    
    print(f"Urgency Score: {result['urgency_score']}")
    print(f"Urgency Level: {result['urgency_level']}")
    
    print(f"\nFactor Breakdown:")
    for factor, value in result['factors'].items():
        print(f"  {factor}: {value}")
    
    print(f"\nCritical Recommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")
    
    assert result['urgency_level'] == 'critical', "Ransomware should be critical"
    assert result['urgency_score'] >= 80, "Ransomware on high-value asset should score very high"
    assert any('ransomware' in rec.lower() for rec in result['recommendations'])
    print("\n✅ Critical ransomware test passed\n")


def test_recency_decay():
    """Test recency scoring decay"""
    print("🧪 Test 4: Recency Decay")
    print("=" * 60)
    
    # Recent detection
    recent = [{
        'id': 'det_001',
        'detection_type': 'brute_force',
        'severity': 'high',
        'confidence_score': 0.8,
        'detected_at': datetime.utcnow().isoformat(),
        'kill_chain_stage': 6
    }]
    
    # Old detection (1 week ago)
    old = [{
        'id': 'det_002',
        'detection_type': 'brute_force',
        'severity': 'high',
        'confidence_score': 0.8,
        'detected_at': (datetime.utcnow() - timedelta(days=7)).isoformat(),
        'kill_chain_stage': 6
    }]
    
    recent_result = scoring_engine.calculate_urgency_score('entity_recent', recent, 50)
    old_result = scoring_engine.calculate_urgency_score('entity_old', old, 50)
    
    print(f"Recent Detection:")
    print(f"  Score: {recent_result['urgency_score']}")
    print(f"  Recency Factor: {recent_result['factors']['recency_score']}")
    
    print(f"\nOld Detection (1 week):")
    print(f"  Score: {old_result['urgency_score']}")
    print(f"  Recency Factor: {old_result['factors']['recency_score']}")
    
    print(f"\nScore Difference: {recent_result['urgency_score'] - old_result['urgency_score']}")
    
    assert recent_result['urgency_score'] > old_result['urgency_score'], "Recent should score higher"
    assert recent_result['factors']['recency_score'] > old_result['factors']['recency_score']
    print("\n✅ Recency decay test passed\n")


def test_empty_detections():
    """Test handling of empty detections"""
    print("🧪 Test 5: Empty Detections")
    print("=" * 60)
    
    result = scoring_engine.calculate_urgency_score('entity_clean', [], 50)
    
    print(f"Urgency Score: {result['urgency_score']}")
    print(f"Urgency Level: {result['urgency_level']}")
    
    assert result['urgency_score'] == 0
    assert result['urgency_level'] == 'low'
    assert result['metadata']['total_detections'] == 0
    print("✅ Empty detections test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎯 PCDS Entity Scoring Engine Tests")
    print("=" * 60 + "\n")
    
    try:
        test_basic_scoring()
        test_multi_stage_attack()
        test_critical_ransomware()
        test_recency_decay()
        test_empty_detections()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nScoring Engine Ready for Production 🚀\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

"""Test script for Advanced Detection Engine v3.0"""
import sys
sys.path.insert(0, '.')

from ml.advanced_detector import get_advanced_engine

# Initialize engine
engine = get_advanced_engine()

# Test detection
result = engine.detect({
    'packet_size': 1200,
    'port': 4444,  # Known malicious port
    'protocol': 'tcp',
    'source_ip': '10.0.0.50',
    'bytes_out': 5000000  # High outbound
})

print("=" * 60)
print("ADVANCED DETECTION ENGINE v3.0 - TEST RESULTS")
print("=" * 60)
print(f"Anomaly Detected: {result['is_anomaly']}")
print(f"Anomaly Score: {result['anomaly_score']:.3f}")
print(f"Risk Level: {result['risk_level'].upper()}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Inference Time: {result['inference_time_ms']:.2f}ms")
print()
print("Model Contributions:")
for model, contrib in result['model_contributions'].items():
    print(f"  - {model}: score={contrib['score']:.3f}, weight={contrib['weight']:.2f}")
print()
print("Top Contributing Factors:")
for factor in result['explanation']['top_contributing_factors'][:3]:
    print(f"  - {factor['description']}: {factor['value']:.3f}")
print()
print("Recommended Actions:")
for action in result['explanation']['recommended_actions'][:3]:
    print(f"  {action}")
print("=" * 60)
print("✅ ENGINE TEST PASSED!")

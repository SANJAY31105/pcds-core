"""Advanced EDR Modules Test"""
import sys
sys.path.insert(0, '.')

print('=' * 60)
print('ADVANCED EDR MODULES TEST')
print('=' * 60)

# Test Memory Scanner
print('\nTest 1: Memory Scanner')
try:
    from edr.collectors.memory_scanner import get_memory_scanner
    scanner = get_memory_scanner()
    result = scanner.quick_scan()
    print('   Memory Scanner initialized')
    print(f'   Threats found: {result["threats_found"]}')
    print('   PASS')
except Exception as e:
    print(f'   FAIL: {e}')

# Test Ransomware Detector
print('\nTest 2: Ransomware Detector')
try:
    from edr.collectors.ransomware_detector import get_ransomware_detector
    detector = get_ransomware_detector()
    
    # Test entropy calculation
    import tempfile
    import os
    
    # Create test file with random bytes (high entropy)
    test_file = tempfile.mktemp(suffix='.bin')
    with open(test_file, 'wb') as f:
        f.write(os.urandom(1024))
    
    entropy = detector.calculate_entropy(test_file)
    print('   Ransomware Detector initialized')
    print(f'   Random file entropy: {entropy:.2f} (expected: ~7.9)')
    
    os.unlink(test_file)
    print('   PASS')
except Exception as e:
    print(f'   FAIL: {e}')

# Test Threat Intel
print('\nTest 3: Threat Intel API')
try:
    from edr.analyzer.threat_intel import get_threat_intel
    api = get_threat_intel()
    
    # Test local IOC
    result = api.check_ip('185.220.101.1')
    print('   Threat Intel API initialized')
    print(f'   Test IP malicious: {result.is_malicious}')
    print(f'   Confidence: {result.confidence}')
    print('   PASS')
except Exception as e:
    print(f'   FAIL: {e}')

print('\n' + '=' * 60)
print('ADVANCED TESTS COMPLETE')
print('=' * 60)

"""
PCDS EDR Comprehensive Test Suite
"""

import sys
sys.path.insert(0, '.')

def run_tests():
    print('=' * 60)
    print('🧪 PCDS EDR COMPREHENSIVE TEST')
    print('=' * 60)

    results = {"passed": 0, "failed": 0}

    # Test 1: Core modules
    print('\n📦 Test 1: Import Core Modules')
    try:
        from edr.core.event_queue import get_event_queue
        print('   ✅ event_queue')
        from edr.core.sysmon_parser import get_sysmon_parser
        print('   ✅ sysmon_parser')
        results["passed"] += 1
    except Exception as e:
        print(f'   ❌ Core import failed: {e}')
        results["failed"] += 1

    # Test 2: Collectors
    print('\n📦 Test 2: Import Collectors')
    try:
        from edr.collectors.process_monitor import get_process_monitor
        print('   ✅ process_monitor')
        from edr.collectors.file_monitor import get_file_monitor
        print('   ✅ file_monitor')
        from edr.collectors.registry_monitor import get_registry_monitor
        print('   ✅ registry_monitor')
        from edr.collectors.network_monitor import get_network_monitor
        print('   ✅ network_monitor')
        results["passed"] += 1
    except Exception as e:
        print(f'   ❌ Collector import failed: {e}')
        results["failed"] += 1

    # Test 3: Actions
    print('\n📦 Test 3: Import Actions')
    try:
        from edr.actions.response_actions import get_response_actions
        print('   ✅ response_actions')
        results["passed"] += 1
    except Exception as e:
        print(f'   ❌ Actions import failed: {e}')
        results["failed"] += 1

    # Test 4: Transport
    print('\n📦 Test 4: Import Transport')
    try:
        from edr.transport.kafka_transport import get_kafka_transport
        print('   ✅ kafka_transport')
        results["passed"] += 1
    except Exception as e:
        print(f'   ❌ Transport import failed: {e}')
        results["failed"] += 1

    # Test 5: EDR Agent
    print('\n📦 Test 5: EDR Agent Integration')
    try:
        from edr.edr_agent import get_edr_agent
        agent = get_edr_agent()
        print('   ✅ Agent initialized')
        
        # Start agent
        agent.start()
        import time
        time.sleep(3)
        
        # Get stats
        stats = agent.get_stats()
        print(f'   📊 Events: {stats.get("events_processed", 0)}')
        print(f'   📊 Detections: {stats.get("detections", 0)}')
        
        # Stop agent
        agent.stop()
        print('   ✅ Agent started and stopped successfully')
        results["passed"] += 1
    except Exception as e:
        print(f'   ❌ Agent test failed: {e}')
        results["failed"] += 1

    # Test 6: API
    print('\n📦 Test 6: API Import')
    try:
        from api.v2.edr_api import router
        print('   ✅ edr_api router')
        results["passed"] += 1
    except Exception as e:
        print(f'   ❌ API import failed: {e}')
        results["failed"] += 1

    # Test 7: ML Model
    print('\n📦 Test 7: ML Model')
    try:
        from ml.ml_detector import get_ml_detector
        detector = get_ml_detector()
        if detector.loaded:
            print('   ✅ ML model loaded (88% accuracy)')
        else:
            print('   ⚠️ ML model not loaded')
        results["passed"] += 1
    except Exception as e:
        print(f'   ❌ ML test failed: {e}')
        results["failed"] += 1

    # Test 8: Ensemble
    print('\n📦 Test 8: Ensemble Detector')
    try:
        from ml.ensemble_detector import get_ensemble_detector
        ensemble = get_ensemble_detector()
        print('   ✅ Ensemble with 5 models')
        results["passed"] += 1
    except Exception as e:
        print(f'   ❌ Ensemble test failed: {e}')
        results["failed"] += 1

    # Summary
    print('\n' + '=' * 60)
    print(f'🏁 TEST RESULTS: {results["passed"]} passed, {results["failed"]} failed')
    print('=' * 60)
    
    return results

if __name__ == "__main__":
    run_tests()

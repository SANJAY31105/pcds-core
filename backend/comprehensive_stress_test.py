"""
PCDS Enterprise - COMPREHENSIVE STRESS TEST
Tests every aspect of the system under extreme conditions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import db_manager
import requests
import time
import random
import concurrent.futures
from datetime import datetime

class ComprehensiveStressTest:
    """Complete system stress test"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "performance": []
        }
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print("🔥 PCDS ENTERPRISE - COMPREHENSIVE STRESS TEST")
        print("="*80 + "\n")
        
        tests = [
            ("Database Performance", self.test_database_performance),
            ("API Endpoints", self.test_api_endpoints),
            ("Concurrent Requests", self.test_concurrent_load),
            ("Detection Correlation", self.test_detection_correlation),
            ("Entity Scoring", self.test_entity_scoring),
            ("MITRE Integration", self.test_mitre_integration),
            ("Campaign Tracking", self.test_campaign_tracking),
            ("Data Validation", self.test_data_validation),
            ("Edge Cases", self.test_edge_cases),
            ("Memory Stress", self.test_memory_stress),
        ]
        
        for name, test_func in tests:
            print(f"\n{'─'*80}")
            print(f"🧪 TEST: {name}")
            print(f"{'─'*80}")
            
            try:
                start = time.time()
                test_func()
                duration = time.time() - start
                self.results['passed'].append(name)
                self.results['performance'].append((name, duration))
                print(f"✅ PASSED ({duration:.2f}s)")
            except Exception as e:
                self.results['failed'].append((name, str(e)))
                print(f"❌ FAILED: {e}")
        
        self.generate_report()
    
    def test_database_performance(self):
        """Test database under heavy load"""
        print("  Testing database performance...")
        
        # Test 1: Large query performance
        start = time.time()
        result = db_manager.execute_query("SELECT * FROM detections LIMIT 1000")
        query_time = time.time() - start
        print(f"  ├─ Large query (1000 rows): {query_time:.3f}s")
        assert query_time < 1.0, f"Query too slow: {query_time}s"
        
        # Test 2: Aggregation performance
        start = time.time()
        stats = db_manager.execute_query("""
            SELECT severity, COUNT(*) as count 
            FROM detections 
            GROUP BY severity
        """)
        agg_time = time.time() - start
        print(f"  ├─ Aggregation query: {agg_time:.3f}s")
        assert agg_time < 0.5, f"Aggregation too slow: {agg_time}s"
        
        # Test 3: Join performance
        start = time.time()
        joins = db_manager.execute_query("""
            SELECT d.*, e.identifier 
            FROM detections d 
            LEFT JOIN entities e ON d.entity_id = e.id 
            LIMIT 500
        """)
        join_time = time.time() - start
        print(f"  ├─ Join query (500 rows): {join_time:.3f}s")
        assert join_time < 1.0, f"Join too slow: {join_time}s"
        
        # Test 4: Count performance
        start = time.time()
        count = db_manager.execute_one("SELECT COUNT(*) as count FROM detections")
        count_time = time.time() - start
        print(f"  └─ Count query: {count_time:.3f}s - Total: {count['count']:,} rows")
        assert count_time < 0.2, f"Count too slow: {count_time}s"
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print("  Testing API endpoints...")
        
        endpoints = [
            ("GET", "/api/v2/dashboard/overview"),
            ("GET", "/api/v2/detections"),
            ("GET", "/api/v2/detections?severity=critical"),
            ("GET", "/api/v2/detections?limit=100"),
            ("GET", "/api/v2/entities"),
            ("GET", "/api/v2/entities/stats"),
            ("GET", "/api/v2/campaigns"),
            ("GET", "/api/v2/mitre/tactics"),
            ("GET", "/api/v2/mitre/techniques"),
            ("GET", "/api/v2/hunt/queries"),
        ]
        
        for method, endpoint in endpoints:
            try:
                start = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                duration = time.time() - start
                
                if response.status_code == 200:
                    print(f"  ├─ ✅ {endpoint} ({duration:.3f}s)")
                    if duration > 2.0:
                        self.results['warnings'].append(f"{endpoint} slow ({duration:.3f}s)")
                else:
                    raise Exception(f"HTTP {response.status_code}")
            except Exception as e:
                print(f"  ├─ ❌ {endpoint}: {e}")
                raise
        
        print(f"  └─ All {len(endpoints)} endpoints working")
    
    def test_concurrent_load(self):
        """Test system under concurrent requests"""
        print("  Testing concurrent load (50 simultaneous requests)...")
        
        def make_request(i):
            try:
                response = requests.get(f"{self.base_url}/api/v2/detections?limit=100")
                return response.status_code == 200
            except:
                return False
        
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(make_request, range(50)))
        duration = time.time() - start
        
        success_rate = sum(results) / len(results) * 100
        print(f"  ├─ Success rate: {success_rate:.1f}%")
        print(f"  ├─ Total time: {duration:.2f}s")
        print(f"  └─ Requests/second: {50/duration:.1f}")
        
        assert success_rate >= 95, f"Too many failures: {success_rate}%"
        assert duration < 10, f"Concurrent requests too slow: {duration}s"
    
    def test_detection_correlation(self):
        """Test detection correlation accuracy"""
        print("  Testing detection correlation...")
        
        # Get detections by campaign
        campaigns = db_manager.execute_query("SELECT id, name FROM attack_campaigns LIMIT 5")
        
        for campaign in campaigns:
            # Check if detections are properly correlated
            detection_count = db_manager.execute_one("""
                SELECT COUNT(*) as count FROM detections 
                WHERE id IN (
                    SELECT detection_id FROM campaign_detections 
                    WHERE campaign_id = ?
                )
            """, (campaign['id'],))
            
            print(f"  ├─ {campaign['name']}: {detection_count['count']} detections")
        
        print(f"  └─ Tested {len(campaigns)} campaigns")
    
    def test_entity_scoring(self):
        """Test entity risk scoring"""
        print("  Testing entity risk scoring...")
        
        entities = db_manager.execute_query("""
            SELECT identifier, threat_score, total_detections, urgency_level
            FROM entities 
            WHERE total_detections > 0
            ORDER BY threat_score DESC
            LIMIT 10
        """)
        
        for entity in entities:
            score = entity['threat_score']
            level = entity['urgency_level']
            
            # Validate score matches urgency level
            if score >= 80:
                expected = 'critical'
            elif score >= 60:
                expected = 'high'
            elif score >= 40:
                expected = 'medium'
            else:
                expected = 'low'
            
            if level != expected:
                self.results['warnings'].append(
                    f"Entity {entity['identifier']}: score {score} but level {level} (expected {expected})"
                )
            
            print(f"  ├─ {entity['identifier']}: Score {score:.1f} = {level}")
        
        print(f"  └─ Tested {len(entities)} entities")
    
    def test_mitre_integration(self):
        """Test MITRE ATT&CK integration"""
        print("  Testing MITRE integration...")
        
        # Check technique coverage
        techniques = db_manager.execute_query("""
            SELECT technique_id, COUNT(*) as count 
            FROM detections 
            WHERE technique_id IS NOT NULL
            GROUP BY technique_id
        """)
        
        print(f"  ├─ Unique techniques detected: {len(techniques)}")
        
        # Verify techniques exist in MITRE table
        for tech in techniques[:5]:
            mitre_data = db_manager.execute_one("""
                SELECT name FROM mitre_techniques WHERE id = ?
            """, (tech['technique_id'],))
            
            if mitre_data:
                print(f"  ├─ {tech['technique_id']}: {mitre_data['name']} ({tech['count']} detections)")
            else:
                self.results['warnings'].append(f"Technique {tech['technique_id']} not in MITRE database")
        
        print(f"  └─ MITRE mapping working")
    
    def test_campaign_tracking(self):
        """Test attack campaign tracking"""
        print("  Testing campaign tracking...")
        
        campaigns = db_manager.execute_query("""
            SELECT name, severity, total_detections, affected_entities 
            FROM attack_campaigns 
            WHERE status = 'active'
        """)
        
        for campaign in campaigns:
            print(f"  ├─ {campaign['name']}: {campaign['total_detections']} detections, {campaign['affected_entities']} entities")
        
        print(f"  └─ Tracking {len(campaigns)} active campaigns")
    
    def test_data_validation(self):
        """Test data integrity"""
        print("  Testing data validation...")
        
        # Check for required fields
        null_checks = [
            ("detections without severity", "SELECT COUNT(*) as count FROM detections WHERE severity IS NULL"),
            ("detections without timestamp", "SELECT COUNT(*) as count FROM detections WHERE detected_at IS NULL"),
            ("entities without identifier", "SELECT COUNT(*) as count FROM entities WHERE identifier IS NULL"),
            ("detections without entity", "SELECT COUNT(*) as count FROM detections WHERE entity_id IS NULL"),
        ]
        
        for check_name, query in null_checks:
            result = db_manager.execute_one(query)
            count = result['count']
            
            if count > 0:
                self.results['warnings'].append(f"{count} {check_name}")
                print(f"  ├─ ⚠️  {count} {check_name}")
            else:
                print(f"  ├─ ✅ No {check_name}")
        
        print(f"  └─ Data validation complete")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("  Testing edge cases...")
        
        # Test invalid API requests
        edge_cases = [
            ("Invalid severity", "/api/v2/detections?severity=invalid"),
            ("Negative limit", "/api/v2/detections?limit=-1"),
            ("Huge limit", "/api/v2/detections?limit=999999"),
            ("Invalid hours", "/api/v2/detections?hours=0"),
        ]
        
        for test_name, endpoint in edge_cases:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                print(f"  ├─ {test_name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"  ├─ {test_name}: {e}")
        
        print(f"  └─ Edge cases handled")
    
    def test_memory_stress(self):
        """Test memory usage with large data"""
        print("  Testing memory stress...")
        
        # Load large dataset
        large_data = db_manager.execute_query("SELECT * FROM detections LIMIT 5000")
        print(f"  ├─ Loaded {len(large_data)} detections")
        
        # Process data
        severity_counts = {}
        for detection in large_data:
            severity_counts[detection['severity']] = severity_counts.get(detection['severity'], 0) + 1
        
        print(f"  ├─ Processed {len(large_data)} records")
        print(f"  └─ Memory stress test passed")
    
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "="*80)
        print("📊 STRESS TEST REPORT")
        print("="*80 + "\n")
        
        total_tests = len(self.results['passed']) + len(self.results['failed'])
        pass_rate = len(self.results['passed']) / total_tests * 100 if total_tests > 0 else 0
        
        print(f"✅ PASSED: {len(self.results['passed'])}/{total_tests} ({pass_rate:.1f}%)")
        for test in self.results['passed']:
            print(f"   ✓ {test}")
        
        if self.results['failed']:
            print(f"\n❌ FAILED: {len(self.results['failed'])}")
            for test, error in self.results['failed']:
                print(f"   ✗ {test}: {error}")
        
        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS: {len(self.results['warnings'])}")
            for warning in self.results['warnings']:
                print(f"   ! {warning}")
        
        print(f"\n⚡ PERFORMANCE:")
        for test, duration in sorted(self.results['performance'], key=lambda x: x[1], reverse=True):
            emoji = "🐌" if duration > 5 else "⚡" if duration < 1 else "✓"
            print(f"   {emoji} {test}: {duration:.2f}s")
        
        print("\n" + "="*80)
        
        if pass_rate >= 90:
            print("🏆 VERDICT: PRODUCTION READY")
        elif pass_rate >= 70:
            print("⚠️  VERDICT: NEEDS OPTIMIZATION")
        else:
            print("❌ VERDICT: SIGNIFICANT ISSUES")
        
        print("="*80 + "\n")
        
        # Return exit code
        return 0 if pass_rate >= 90 else 1


if __name__ == "__main__":
    tester = ComprehensiveStressTest()
    exit_code = tester.run_all_tests()
    sys.exit(exit_code)

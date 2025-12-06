"""
Test PCDS Enterprise API v2 Endpoints
Quick validation of all major endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(name, url):
    """Test an API endpoint"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}")
            return response.json()
        else:
            print(f"❌ {name} - Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ {name} - Error: {str(e)[:50]}")
        return None

def main():
    print("\n" + "="*60)
    print("🧪 PCDS Enterprise API v2 Test Suite")
    print("="*60 + "\n")
    
    # Test root endpoints
    print("📡 Testing Core Endpoints:")
    test_endpoint("Root", f"{BASE_URL}/")
    test_endpoint("Health", f"{BASE_URL}/health")
    test_endpoint("API Status", f"{BASE_URL}/api/v2/status")
    
    # Test dashboard
    print("\n📊 Testing Dashboard:")
    dashboard = test_endpoint("Dashboard Overview", f"{BASE_URL}/api/v2/dashboard/overview?hours=24")
    if dashboard:
        print(f"   - Entities: {dashboard.get('entities', {}).get('total', 0)}")
        print(f"   - Detections: {dashboard.get('detections', {}).get('total', 0)}")
        print(f"   - MITRE Coverage: {dashboard.get('mitre', {}).get('coverage_percentage', 0)}%")
    
    # Test entities
    print("\n👥 Testing Entities API:")
    entities = test_endpoint("Entities List", f"{BASE_URL}/api/v2/entities?limit=10")
    stats = test_endpoint("Entity Stats", f"{BASE_URL}/api/v2/entities/stats/overview")
    
    # Test detections
    print("\n🎯 Testing Detections API:")
    detections = test_endpoint("Detections List", f"{BASE_URL}/api/v2/detections?limit=10")
    severity = test_endpoint("Severity Breakdown", f"{BASE_URL}/api/v2/detections/stats/severity-breakdown")
    
    # Test campaigns
    print("\n🔗 Testing Campaigns API:")
    campaigns = test_endpoint("Campaigns List", f"{BASE_URL}/api/v2/campaigns?limit=10")
    
    # Test investigations
    print("\n🔍 Testing Investigations API:")
    investigations = test_endpoint("Investigations List", f"{BASE_URL}/api/v2/investigations?limit=10")
    
    # Test hunt
    print("\n🎯 Testing Hunt API:")
    hunt_queries = test_endpoint("Hunt Queries", f"{BASE_URL}/api/v2/hunt/queries")
    
    # Test MITRE
    print("\n🛡️ Testing MITRE API:")
    tactics = test_endpoint("MITRE Tactics", f"{BASE_URL}/api/v2/mitre/tactics")
    coverage = test_endpoint("MITRE Coverage", f"{BASE_URL}/api/v2/mitre/stats/coverage")
    if coverage:
        print(f"   - Techniques Detected: {coverage.get('covered_techniques', 0)}/{coverage.get('total_techniques', 0)}")
        print(f"   - Coverage: {coverage.get('coverage_percentage', 0)}%")
    
    heatmap = test_endpoint("MITRE Heatmap", f"{BASE_URL}/api/v2/mitre/matrix/heatmap?hours=24")
    
    print("\n" + "="*60)
    print("✅ API v2 Test Suite Complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

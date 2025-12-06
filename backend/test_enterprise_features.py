"""
Test script for PCDS Enterprise Phase 1-3 features
Tests: Redis caching, JWT auth, RBAC, Celery tasks
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

def test_redis_cache():
    """Test Redis cache client"""
    print("\n🧪 Testing Redis Cache Client...")
    try:
        from cache.redis_client import cache_client
        
        # Test basic operations
        cache_client.set("test_key", {"value": "test"}, ttl=60)
        result = cache_client.get("test_key")
        
        assert result == {"value": "test"}, "Cache get/set failed"
        print("   ✅ Cache get/set working")
        
        # Test delete
        cache_client.delete("test_key")
        result = cache_client.get("test_key")
        assert result is None, "Cache delete failed"
        print("   ✅ Cache delete working")
        
        # Test ping
        if cache_client.ping():
            print("   ✅ Redis connection: CONNECTED")
        else:
            print("   ⚠️  Redis connection: NOT CONNECTED (start with: docker-compose up -d redis)")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_jwt_auth():
    """Test JWT authentication"""
    print("\n🧪 Testing JWT Authentication...")
    try:
        from auth.jwt import jwt_manager
        
        # Test token creation
        tokens = jwt_manager.create_token_pair(
            user_id="test@college.edu",
            role="admin"
        )
        
        assert "access_token" in tokens, "Access token not created"
        assert "refresh_token" in tokens, "Refresh token not created"
        print("   ✅ Token pair creation working")
        
        # Test token verification
        payload = jwt_manager.verify_access_token(tokens["access_token"])
        assert payload is not None, "Token verification failed"
        assert payload["sub"] == "test@college.edu", "User ID mismatch"
        assert payload["role"] == "admin", "Role mismatch"
        print("   ✅ Token verification working")
        
        # Test password hashing
        hashed = jwt_manager.hash_password("test_password")
        assert jwt_manager.verify_password("test_password", hashed), "Password verification failed"
        print("   ✅ Password hashing working")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_rbac():
    """Test RBAC system"""
    print("\n🧪 Testing RBAC System...")
    try:
        from auth.rbac import Role, Permission, rbac_manager
        
        # Test admin permissions
        admin_perms = rbac_manager.get_permissions(Role.ADMIN)
        assert Permission.MANAGE_USERS in admin_perms, "Admin missing manage users permission"
        print(f"   ✅ Admin has {len(admin_perms)} permissions")
        
        # Test analyst permissions
        analyst_perms = rbac_manager.get_permissions(Role.ANALYST)
        assert Permission.WRITE_DETECTIONS in analyst_perms, "Analyst missing write permission"
        assert Permission.MANAGE_USERS not in analyst_perms, "Analyst should not have manage users"
        print(f"   ✅ Analyst has {len(analyst_perms)} permissions")
        
        # Test viewer permissions
        viewer_perms = rbac_manager.get_permissions(Role.VIEWER)
        assert Permission.READ_DETECTIONS in viewer_perms, "Viewer missing read permission"
        assert Permission.WRITE_DETECTIONS not in viewer_perms, "Viewer should not have write"
        print(f"   ✅ Viewer has {len(viewer_perms)} permissions")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_celery_tasks():
    """Test Celery task definitions"""
    print("\n🧪 Testing Celery Tasks...")
    try:
        from tasks.celery_app import celery_app
        from tasks import queue_entity_scoring, queue_campaign_correlation
        
        # Check registered tasks
        task_names = [
            'tasks.calculate_entity_risk',
            'tasks.correlate_campaign',
            'tasks.analyze_threat_intelligence',
            'tasks.generate_report',
            'tasks.cleanup_old_data'
        ]
        
        for task_name in task_names:
            assert task_name in celery_app.tasks, f"Task {task_name} not registered"
        
        print(f"   ✅ All 5 background tasks registered")
        print(f"   ✅ Total Celery tasks: {len(celery_app.tasks)}")
        print("   ℹ️  To run worker: celery -A tasks.celery_app worker --loglevel=info")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_audit_logging():
    """Test audit logging"""
    print("\n🧪 Testing Audit Logging...")
    try:
        from middleware.audit_log import audit_logger
        
        # Note: Requires Redis connection
        print("   ✅ Audit logger imported successfully")
        print("   ℹ️  Audit logging requires Redis to be running")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 PCDS ENTERPRISE - PHASE 1-3 TESTING")
    print("=" * 60)
    
    results = {
        "Redis Cache": test_redis_cache(),
        "JWT Auth": test_jwt_auth(),
        "RBAC System": test_rbac(),
        "Celery Tasks": test_celery_tasks(),
        "Audit Logging": test_audit_logging()
    }
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name:20} {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Enterprise features ready to use!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)
    print("1. Start Redis: docker-compose up -d redis")
    print("2. Test caching: curl http://localhost:8000/api/v2/dashboard/overview")
    print("3. Start Celery worker: celery -A tasks.celery_app worker --loglevel=info")
    print("4. Install frontend deps: cd frontend && npm install")
    print("=" * 60)


if __name__ == "__main__":
    main()

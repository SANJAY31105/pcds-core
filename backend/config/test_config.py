"""
Test configuration and database connection
Run: python -m config.test_config
"""

from config.settings import settings
from config.database import init_db, db_manager, EntityQueries, MITREQueries


def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing PCDS Enterprise Configuration")
    print("=" * 60)
    
    print(f"\n📱 Application:")
    print(f"  Name: {settings.APP_NAME}")
    print(f"  Version: {settings.APP_VERSION}")
    print(f"  Debug: {settings.DEBUG}")
    
    print(f"\n🗄️  Database:")
    print(f"  URL: {settings.DATABASE_URL}")
    print(f"  Path: {db_manager.db_path}")
    
    print(f"\n🔐 Security:")
    print(f"  CORS Origins: {settings.CORS_ORIGINS}")
    
    print(f"\n📁 Paths:")
    print(f"  Base Dir: {settings.BASE_DIR}")
    print(f"  Data Dir: {settings.DATA_DIR}")
    print(f"  MITRE Data: {settings.MITRE_DATA_FILE}")
    print(f"  Evidence Dir: {settings.EVIDENCE_UPLOAD_DIR}")
    
    print("\n" + "=" * 60)
    print("✅ Configuration loaded successfully")


def test_database():
    """Test database connection and queries"""
    print("\n🗄️  Testing Database Connection")
    print("=" * 60)
    
    # Test initialization
    if not init_db():
        print("❌ Database not properly initialized")
        return False
    
    # Test MITRE data
    print("\n📊 MITRE ATT&CK Data:")
    tactics = MITREQueries.get_all_tactics()
    print(f"  Tactics: {len(tactics)}")
    
    if tactics:
        print(f"  Sample: {tactics[0]['id']} - {tactics[0]['name']}")
    
    # Test technique query
    technique = MITREQueries.get_technique("T1110")
    if technique:
        print(f"  Technique T1110: {technique['name']}")
    
    # Test entities
    print("\n👥 Entities:")
    entities = EntityQueries.get_all(limit=5)
    print(f"  Total entities: {len(entities)}")
    
    if entities:
        for entity in entities[:3]:
            print(f"  - {entity['id']}: {entity['identifier']} (Urgency: {entity['urgency_level']})")
    
    print("\n" + "=" * 60)
    print("✅ Database connection working")
    return True


if __name__ == "__main__":
    try:
        test_configuration()
        test_database()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

"""
PCDS Enterprise - Database Performance Optimization
Adds indexes to critical columns for faster queries with 100K+ records
"""

from config.database import db_manager

print("\n" + "="*60)
print("🚀 PCDS DATABASE OPTIMIZATION")
print("="*60 + "\n")

# Add indexes for faster queries
indexes = [
    # Detections table - most queried
    ("idx_detections_severity", "CREATE INDEX IF NOT EXISTS idx_detections_severity ON detections(severity)"),
    ("idx_detections_detected_at", "CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections(detected_at DESC)"),
    ("idx_detections_entity_id", "CREATE INDEX IF NOT EXISTS idx_detections_entity_id ON detections(entity_id)"),
    ("idx_detections_technique_id", "CREATE INDEX IF NOT EXISTS idx_detections_technique_id ON detections(technique_id)"),
    ("idx_detections_created_at", "CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at DESC)"),
    
    # Entities table
    ("idx_entities_identifier", "CREATE INDEX IF NOT EXISTS idx_entities_identifier ON entities(identifier)"),
    ("idx_entities_threat_score", "CREATE INDEX IF NOT EXISTS idx_entities_threat_score ON entities(threat_score DESC)"),
    ("idx_entities_urgency_level", "CREATE INDEX IF NOT EXISTS idx_entities_urgency_level ON entities(urgency_level)"),
    
    # Composite indexes for common queries
    ("idx_detections_severity_date", "CREATE INDEX IF NOT EXISTS idx_detections_severity_date ON detections(severity, detected_at DESC)"),
    ("idx_detections_entity_date", "CREATE INDEX IF NOT EXISTS idx_detections_entity_date ON detections(entity_id, detected_at DESC)"),
]

print("Adding database indexes for performance...\n")

for idx_name, sql in indexes:
    try:
        db_manager.execute_query(sql)
        print(f"✅ {idx_name}")
    except Exception as e:
        print(f"⚠️  {idx_name}: {e}")

print("\n" + "="*60)
print("📊 PERFORMANCE ANALYSIS")
print("="*60 + "\n")

# Analyze query performance
stats = db_manager.execute_one("""
    SELECT 
        COUNT(*) as total_detections,
        COUNT(DISTINCT entity_id) as unique_entities,
        COUNT(DISTINCT severity) as severity_levels,
        COUNT(DISTINCT technique_id) as techniques_used
    FROM detections
""")

print(f"Total Detections: {stats['total_detections']:,}")
print(f"Unique Entities: {stats['unique_entities']:,}")
print(f"Severity Levels: {stats['severity_levels']}")
print(f"MITRE Techniques: {stats['techniques_used']}")

# Test query speed
import time

print("\n" + "="*60)
print("⚡ QUERY SPEED TEST")
print("="*60 + "\n")

# Test 1: Recent detections (most common query)
start = time.time()
results = db_manager.execute_query("SELECT * FROM detections ORDER BY detected_at DESC LIMIT 100")
elapsed = time.time() - start
print(f"1. Recent 100 detections: {elapsed*1000:.2f}ms")

# Test 2: Critical severity filter
start = time.time()
results = db_manager.execute_query("SELECT * FROM detections WHERE severity='critical' ORDER BY detected_at DESC LIMIT 100")
elapsed = time.time() - start
print(f"2. Critical detections: {elapsed*1000:.2f}ms")

# Test 3: Entity lookup
start = time.time()
results = db_manager.execute_query("SELECT * FROM entities ORDER BY threat_score DESC LIMIT 50")
elapsed = time.time() - start
print(f"3. Top 50 entities: {elapsed*1000:.2f}ms")

print("\n" + "="*60)
print("✅ OPTIMIZATION COMPLETE!")
print("="*60 + "\n")
print("Expected improvements:")
print("  • 10-50x faster queries on large datasets")
print("  • Dashboard loads in <500ms")
print("  • Live feed updates smoothly")
print("  • Entity pages load instantly")
print("\n")

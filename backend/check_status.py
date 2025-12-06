"""Quick status check for AI Attack Simulation"""
from config.database import db_manager

# Get counts
detections = db_manager.execute_one("SELECT COUNT(*) as c FROM detections")
entities = db_manager.execute_one("SELECT COUNT(*) as c FROM entities")
critical = db_manager.execute_one("SELECT COUNT(*) as c FROM detections WHERE severity='critical'")
high = db_manager.execute_one("SELECT COUNT(*) as c FROM detections WHERE severity='high'")

print("\n" + "="*60)
print("📊 PCDS ENTERPRISE - CURRENT STATUS")
print("="*60)
print(f"\nTotal Detections: {detections['c']:,}")
print(f"Total Entities: {entities['c']:,}")
print(f"Critical Threats: {critical['c']:,}")
print(f"High Threats: {high['c']:,}")
print("\n" + "="*60)
print("✅ Data is loaded and ready!")
print("🌐 Refresh your browser to see all attacks!")
print("="*60 + "\n")

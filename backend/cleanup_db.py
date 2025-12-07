import sqlite3
from datetime import datetime

# Connect to database
conn = sqlite3.connect('pcds_enterprise.db')
cursor = conn.cursor()

# Delete detections older than today
today = datetime.now().strftime('%Y-%m-%d')
cursor.execute(f"DELETE FROM detections WHERE date(detected_at) < date('{today}')")
deleted = cursor.rowcount
conn.commit()

# Also clean up entities if empty
cursor.execute("DELETE FROM entities WHERE id NOT IN (SELECT DISTINCT entity_id FROM detections WHERE entity_id IS NOT NULL)")
entities_deleted = cursor.rowcount
conn.commit()

conn.close()

print(f"✅ Deleted {deleted} old detections")
print(f"✅ Cleaned up {entities_deleted} orphan entities")
print(f"📅 Only keeping data from {today}")

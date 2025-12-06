from config.database import db_manager

# Check detections in database
dets = db_manager.execute_query('SELECT id, detection_type, severity, title FROM detections LIMIT 10')
print(f'✅ Detections in database: {len(dets)}')
for d in dets:
    print(f"  {d['severity'].upper()}: {d['title']}")

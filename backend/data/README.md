# PCDS Enterprise Database

## ✅ Step 1 Complete: Database Foundation

### Files Created

1. **`schema.sql`** - Complete database schema
   - 15+ tables (entities, detections, campaigns, investigations, MITRE, hunt queries)
   - Indexes for performance
   - Foreign key relationships

2. **`mitre_attack_full.json`** - MITRE ATT&CK Framework Data
   - 12 tactics (TA0001 - TA0040)
   - 40+ techniques with detailed metadata
   - Detection type mappings

3. **`init_db.py`** - Database initialization script
   - Creates tables
   - Seeds MITRE data
   - Creates 6 default hunt queries

4. **`pcds_enterprise.db`** - SQLite database (auto-generated)

### Database Statistics

- **Tables**: 15 core tables + indexes
- **MITRE Tactics**: 12
- **MITRE Techniques**: 40+
- **Default Hunt Queries**: 6
  - Credential Theft Activity
  - Lateral Movement Patterns
  - Command & Control Beaconing
  - Data Exfiltration
  - Privilege Escalation Attempts
  - Ransomware Indicators

### Re-initialize Database

```bash
cd backend
python data/init_db.py
```

### Next Steps

**Step 2**: Create backend configuration and database connection
- `config/settings.py` - Environment configuration
- `config/database.py` - SQLite/PostgreSQL connection manager

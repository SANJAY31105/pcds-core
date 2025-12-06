import sqlite3
import uuid
from datetime import datetime
import sys
sys.path.append('.')
from auth.jwt import jwt_manager

# Create users table
conn = sqlite3.connect('pcds_enterprise.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'analyst',
        is_active BOOLEAN DEFAULT 1,
        created_at TEXT,
        last_login TEXT
    )
''')
conn.commit()
print("✅ Users table created!")

# Check if admin exists
cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
if cursor.fetchone()[0] > 0:
    print("⚠️  Admin user already exists")
else:
    # Create admin user
    pw_hash = jwt_manager.hash_password('admin123')
    user_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO users (id, username, email, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, 'admin', 'admin@pcds.local', pw_hash, 'admin', 1, datetime.utcnow().isoformat()))
    conn.commit()
    
    print("✅ Admin user created!")
    print("=" * 60)
    print("LOGIN CREDENTIALS:")
    print("Username: admin")
    print("Password: admin123")
    print("=" * 60)

conn.close()

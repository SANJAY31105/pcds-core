import sqlite3
import uuid
from datetime import datetime

# Create users table and admin user WITHOUT bcrypt (direct approach)
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
    cursor.execute("SELECT username, email, role FROM users WHERE username = 'admin'")
    user = cursor.fetchone()
    print(f"Existing admin: {user[0]} ({user[1]}), role: {user[2]}")
else:
    # Use a pre-computed bcrypt hash for "admin123"
    # This is the bcrypt hash of "admin123" with cost factor 12
    pw_hash = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqnm7bvvJS"
    
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

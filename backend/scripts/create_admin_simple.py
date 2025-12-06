import sqlite3
import uuid
from datetime import datetime

# Simple direct admin creation without bcrypt complications
conn = sqlite3.connect('data/pcds_enterprise.db')
cursor = conn.cursor()

# Check if users exist
cursor.execute("SELECT COUNT(*) FROM users")
count = cursor.fetchone()[0]

if count > 0:
    print(f"⚠️ {count} user(s) already exist. Deleting all users...")
    cursor.execute("DELETE FROM users")
    cursor.execute("DELETE FROM tenants")
    conn.commit()

# Create tenant
tenant_id = str(uuid.uuid4())
cursor.execute("""
    INSERT INTO tenants (id, name, created_at, is_active)
    VALUES (?, ?, ?, ?)
""", (tenant_id, "Default Organization", datetime.now().isoformat(), 1))

# Create admin user with SIMPLE hash using passlib directly
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hash the password
password = "admin123"
hashed_password = pwd_context.hash(password)

admin_id = str(uuid.uuid4())
cursor.execute("""
    INSERT INTO users (id, email, full_name, password_hash, role, tenant_id, created_at, is_active)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (admin_id, "admin@pcds.local", "Super Administrator", hashed_password, "super_admin", tenant_id, datetime.now().isoformat(), 1))

conn.commit()

# Verify
cursor.execute("SELECT email, role FROM users")
users = cursor.fetchall()
print("\n" + "="*60)
print("✅ ADMIN USER CREATED SUCCESSFULLY!")
print("="*60)
print(f"Email: admin@pcds.local")
print(f"Password: {password}")
print(f"Role: super_admin")
print(f"Tenant: {tenant_id}")
print("="*60)
print("\n✅ Users in database:", users)

conn.close()

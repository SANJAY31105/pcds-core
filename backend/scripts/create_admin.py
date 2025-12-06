# PCDS Enterprise - Security Setup Script
# Creates first admin user for enterprise deployment

import sqlite3
import uuid
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.jwt import jwt_manager

def create_admin_user(username: str, email: str, password: str):
    """Create the first admin user"""
    
    # Connect to database
    db_path = "../pcds_enterprise.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if admin already exists
        cursor.execute("SELECT id FROM users WHERE role = 'admin'")
        existing_admin = cursor.fetchone()
        
        if existing_admin:
            print("❌ Admin user already exists!")
            return False
        
        # Hash password
        password_hash = jwt_manager.hash_password(password)
        
        # Create admin user
        user_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO users (id, username, email, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, ?, 'admin', 1, ?)
        """, (user_id, username, email, password_hash, datetime.utcnow().isoformat()))
        
        conn.commit()
        
        print("✅ Admin user created successfully!")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   Role: admin")
        print("\n🔒 SECURITY: Public registration is disabled.")
        print("   Only admins can create new users via the Admin Panel.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("PCDS Enterprise - Admin User Setup")
    print("=" * 60)
    print()
    
    # Get admin credentials
    print("Create your first admin user:")
    username = input("Username: ").strip()
    email = input("Email: ").strip()
    password = input("Password (min 8 chars): ").strip()
    
    if len(password) < 8:
        print("❌ Password must be at least 8 characters!")
        sys.exit(1)
    
    # Create admin
    create_admin_user(username, email, password)

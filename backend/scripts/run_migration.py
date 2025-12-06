import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3

def run_migration():
    conn = sqlite3.connect('data/pcds_enterprise.db')
    cursor = conn.cursor()
    
    # Read and execute SQL file
    with open('data/add_auth_tables.sql', 'r') as f:
        sql_script = f.read()
    
    cursor.executescript(sql_script)
    
    # Add tenant_id columns to existing tables if they don't exist
    tables_to_migrate = ['entities', 'detections', 'investigations', 'campaigns']
    
    for table in tables_to_migrate:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT")
            print(f"Added tenant_id to {table}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"tenant_id already exists in {table}")
            else:
                raise
    
    conn.commit()
    conn.close()
    print("Migration completed successfully!")

if __name__ == "__main__":
    run_migration()

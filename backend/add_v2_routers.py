"""
Quick script to add API v2 router registration to main.py
"""

# Read the current main.py
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if routers are already registered
if 'app.include_router(entities_router' in content:
    print("✅ API v2 routers already registered!")
else:
    # Find the position after CORS middleware
    cors_end = content.find(')\n\n\n# ============= Core API Endpoints =============')
    
    if cors_end == -1:
        cors_end = content.find(')\n\n# ============= Core API Endpoints =============')
    
    if cors_end != -1:
        # Insert router registration code
        router_code = '''\n\n# Register API v2 Routers
if HAS_API_V2:
    print("🔌 Registering API v2 endpoints...")
    app.include_router(entities_router, prefix="/api/v2")
    app.include_router(detections_router, prefix="/api/v2")
    app.include_router(campaigns_router, prefix="/api/v2")
    app.include_router(investigations_router, prefix="/api/v2")
    app.include_router(hunt_router, prefix="/api/v2")
    app.include_router(mitre_router, prefix="/api/v2")
    app.include_router(dashboard_router, prefix="/api/v2")
    print("✅ API v2 endpoints registered (36 total)")
'''
        
        # Insert at the correct position
        new_content = content[:cors_end+1] + router_code + content[cors_end+1:]
        
        # Write back
        with open('main.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("✅ Router registration code added to main.py!")
        print("Server should auto-reload with API v2 endpoints")
    else:
        print("❌ Could not find insertion point in main.py")

import requests
import json

# Test detections API
try:
    response = requests.get('http://localhost:8000/api/v2/detections')
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ API Working!")
        print(f"Total Detections: {data.get('total', 0)}")
        print(f"\nDetections:")
        for d in data.get('detections', [])[:5]:
            print(f"  - {d.get('severity').upper()}: {d.get('title', d.get('description', 'No title'))}")
    else:
        print(f"\n❌ Error: {response.text}")
except Exception as e:
    print(f"\n❌ Exception: {e}")

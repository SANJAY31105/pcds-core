"""Create test validation data"""
import requests
import numpy as np

API_BASE = 'http://localhost:8000/api/v2'

print("Creating test predictions and feedback...")

# Create 5 true positives
for i in range(5):
    r = requests.post(f'{API_BASE}/ml/shadow/predict', json={
        'features': np.random.randn(40).tolist(),
        'source_host': f'validation-tp-{i}'
    })
    pid = r.json().get('prediction_id')
    
    requests.post(f'{API_BASE}/ml/shadow/feedback', json={
        'prediction_id': pid,
        'is_correct': True
    })
    print(f'  ✓ TP: {pid}')

# Create 2 false positives
for i in range(2):
    r = requests.post(f'{API_BASE}/ml/shadow/predict', json={
        'features': np.random.randn(40).tolist(),
        'source_host': f'validation-fp-{i}'
    })
    pid = r.json().get('prediction_id')
    
    requests.post(f'{API_BASE}/ml/shadow/feedback', json={
        'prediction_id': pid,
        'is_correct': False
    })
    print(f'  ✗ FP: {pid}')

# Check validation endpoint
print("\nFetching validation metrics...")
r = requests.get(f'{API_BASE}/ml/shadow/validation')
data = r.json()
print(f"Total reviewed: {data.get('total_reviewed', 0)}")
print(f"Overall metrics: {data.get('overall_metrics', {})}")

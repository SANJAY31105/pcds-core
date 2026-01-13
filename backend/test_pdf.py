"""Test PDF export"""
import requests

# Create report
print("Creating executive report...")
r = requests.post('http://localhost:8000/api/v2/reports/generate', json={'report_type': 'executive'})
data = r.json()
report_id = data.get('report_id')
print(f"Report ID: {report_id}")

# Download PDF
print("Downloading PDF...")
r2 = requests.get(f'http://localhost:8000/api/v2/reports/download/{report_id}/pdf')
print(f"PDF Status: {r2.status_code}")
print(f"Content-Type: {r2.headers.get('Content-Type')}")
print(f"Size: {len(r2.content)} bytes")

if r2.status_code == 200:
    # Save PDF locally
    with open(f'{report_id}.pdf', 'wb') as f:
        f.write(r2.content)
    print(f"✅ PDF saved as {report_id}.pdf")
else:
    print(f"❌ Error: {r2.text[:200]}")

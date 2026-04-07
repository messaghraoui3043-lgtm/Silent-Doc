from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)
with open('ISIC_0024948.jpg', 'rb') as f:
    response = client.post('/predict/skin', files={'file': ('image.jpg', f, 'image/jpeg')}, data={'session_id': 'test'})
    print("Status:", response.status_code)
    print("JSON:", response.json())

from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)
with open('audio/tmp/input.wav', 'rb') as f:
    response = client.post('/predict/voice', files={'file': ('record.webm', f, 'audio/webm')}, data={'session_id': 'test2'})
    print("Status:", response.status_code)
    try:
        print("JSON Error Detail:", response.json()['detail'])
    except:
        print("JSON:", response.json())

from fastapi.testclient import TestClient
from app import app  # your FastAPI instance

client = TestClient(app)

def test_app_home():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'messages': "Hey this is home page for our API"}


def test_app_prediction():
    input_data = {
        "age": 12,
        "weight": 40,
        "height": 1.34,
        "income_lpa": 5,
        "smoker": True,
        "city": "Pune",
        "occupation": "freelancer"
    }
    response = client.post('/predict', json=input_data)  # <-- send JSON body to FastAPI
    assert response.status_code == 200
    assert "prediction" in response.json()  # example: check returned JSON

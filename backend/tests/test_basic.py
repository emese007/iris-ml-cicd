from fastapi.testclient import TestClient
from app.main import app  # important: backend.app.main

client = TestClient(app)


def test_root_status_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_streamlit_payload():
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)
    assert data["prediction"] in [0, 1, 2]





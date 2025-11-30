import requests

API_URL = "http://127.0.0.1:8000/predict"

def test_predict_endpoint():
    file_path = "tests/sample_images/healthy1.jpg"
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(API_URL, files=files)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data

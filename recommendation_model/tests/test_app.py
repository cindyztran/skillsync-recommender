from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_qna_endpoint():
    response = client.post("/qna", json={"question": "What is AI?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert isinstance(response.json()["answer"], str)

def test_recommend_endpoint():
    response = client.post("/recommend", json={"user_id": 1, "skill_id": 101})
    assert response.status_code == 200
    json_data = response.json()
    
    assert "user_id" in json_data
    assert "skill_id" in json_data
    assert "predicted_rating" in json_data
    assert isinstance(json_data["predicted_rating"], float)  # Ensure it’s a float

def test_personalized_learning_plan():
    response = client.post("/personalized-learning-plan", json={"user_id": 1, "goal": "Learn AI"})
    assert response.status_code == 200
    json_data = response.json()
    
    assert "user_id" in json_data
    assert "recommended_skills" in json_data
    assert "learning_plan" in json_data
    assert isinstance(json_data["recommended_skills"], list)  # Ensure it's a list
    assert isinstance(json_data["learning_plan"], str)  # Ensure it's a string
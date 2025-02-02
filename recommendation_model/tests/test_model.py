import torch
from app.model import RecommendationModel

def test_model_initialization():
    num_users = 10
    num_skills = 50
    model = RecommendationModel(num_users, num_skills)
    assert isinstance(model, torch.nn.Module)

def test_model_prediction():
    num_users = 10
    num_skills = 50
    model = RecommendationModel(num_users, num_skills)
    
    user_tensor = torch.tensor([1])
    skill_tensor = torch.tensor([2])
    
    output = model(user_tensor, skill_tensor)
    assert isinstance(output.item(), float)  # Ensure output is a single float value
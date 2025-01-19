from fastapi import FastAPI
from pydantic import BaseModel
import torch
from app.model import RecommendationModel, user_mapping, skill_mapping

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = RecommendationModel(len(user_mapping), len(skill_mapping))
model.load_state_dict(torch.load("models/recommendation_model.pth"))
model.eval()

# Request schema
class RecommendationRequest(BaseModel):
    user_id: int
    skill_id: int

# Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Recommendation API"}

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    user_id = request.user_id
    skill_id = request.skill_id

    if user_id not in user_mapping or skill_id not in skill_mapping:
        return {"error": "User or Skill not found!"}

    user_tensor = torch.tensor([user_mapping[user_id]])
    skill_tensor = torch.tensor([skill_mapping[skill_id]])
    prediction = model(user_tensor, skill_tensor).item()

    return {
        "user_id": user_id,
        "skill_id": skill_id,
        "predicted_rating": round(prediction, 2),
    }
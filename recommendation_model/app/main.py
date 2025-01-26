from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from app.model import RecommendationModel, user_mapping, skill_mapping
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Initialize FastAPI app
app = FastAPI()

# Set OpenAI API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
)

# Load the trained model
model = RecommendationModel(len(user_mapping), len(skill_mapping))
model.load_state_dict(torch.load("models/recommendation_model.pth"))
model.eval()

openai_model = "gpt-4o-mini"


# Request schema
class RecommendationRequest(BaseModel):
    user_id: int
    skill_id: int

# Request models
class QnARequest(BaseModel):
    question: str

class PlanRequest(BaseModel):
    goal: str
    duration_weeks: int

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

@app.post("/qna")
def answer_question(request: QnARequest):
  try: 
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Answer the following question: {request.question}"
            }
        ],
        model=openai_model,
    )
    return {"question": request.question, "answer": response.choices[0].message.content.strip()}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

 
@app.post("/generate-plan")
def generate_plan(request: PlanRequest):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Create a {request.duration_weeks}-week learning plan to achieve the goal: "
                    f"{request.goal}."
                }
            ],
            model=openai_model,
        )
        return {"goal": request.goal, "duration_weeks": request.duration_weeks, "plan":  response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
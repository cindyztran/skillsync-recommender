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
    
class LearningPlanRequest(BaseModel):
    user_id: int
    goal: str

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
    
@app.post("/personalized-learning-plan")
def generate_personalized_plan(request: LearningPlanRequest):
    try:
        # Step 1: PyTorch Skill Recommendation
        user_tensor = torch.tensor([request.user_id])  # Example user input
        skill_ids = list(range(len(skill_mapping)))  # Generate all skill IDs for predictions
        skill_tensor = torch.tensor(skill_ids)  # Create tensor for all possible skills

        # Get predictions from the model
        with torch.no_grad():  # Disable gradient tracking for inference
            predictions = model(user_tensor.repeat(len(skill_tensor)), skill_tensor)
        
        # Sort skills by predicted ratings (descending order)
        k = min(5, predictions.size(0))  # Ensure k does not exceed the number of predictions
        _, top_indices = torch.topk(predictions, k=k)
        print("Skill tensor size:", skill_tensor.size())
        print("Predictions size:", predictions.size())
        print("Top k value:", k)
        recommended_skills = [f"Skill {skill_ids[i]}" for i in top_indices.numpy()]

        # Step 2: GPT Plan Generation
        skill_list = ", ".join(recommended_skills)
        gpt_response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Create a learning plan for the goal '{request.goal}' using these skills: {skill_list}",
                }
            ],
            model=openai_model,
        )
        learning_plan = gpt_response.choices[0].message.content.strip()
        # Step 3: Return Combined Response
        return {
            "user_id": request.user_id,
            "recommended_skills": recommended_skills,
            "learning_plan": learning_plan
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
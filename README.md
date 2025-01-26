# Recommendation Model API

This project demonstrates a recommendation system deployed using FastAPI.

## **Table of Contents**

1. [Features](#features)
2. [Setup](#setup)
3. [Training the Model](#training-the-model)
4. [Running the Server](#running-the-server)
5. [Making API Requests](#making-api-requests)
6. [Files Description](#files-description)

## Features

### • API Endpoints:

- /: Root endpoint to verify the server is running.
- /recommend: Predict skill ratings for a given user and skill.
- /qna: Allows users to ask questions, and OpenAI’s GPT API provides answers.
- /personalized-learning-plan: Combines the PyTorch recommendation system and OpenAI’s GPT API to generate a personalized learning plan for a user. Recommends skills based on user preferences (via PyTorch) and generates a detailed learning plan using GPT.

## Setup

1. Clone the Repository

```
git clone https://github.com/cindyztran/skillsync-recommender.git
cd skillsync-recommender
```

2. Install Dependencies

```
pip install -r requirements.txt
```

3. Verify Project Structure

Ensure your project has the following structure:

```
recommendation_model/
├── app/
│ ├── main.py # FastAPI application
│ ├── model.py # Recommendation model logic
├── recommendation_model.py # Script to train and save the model
├── models/
│ └── recommendation_model.pth # Trained model (generated after training)
├── requirements.txt # Project dependencies
├── README.md # Project documentation
```

## Training the Model

1. Run the Training Script
   Train the recommendation model using:

```
python recommendation_model.py
```

2. Check the Output

- After successful training, the model file **`recommendation_model.pth`** will be saved in the **`models/`** directory.

3. Debugging Tips
   - If the file is not generated, check:
     - That the models/ directory exists.
     - The training script runs without errors.

## Running the Server

1. Start the FastAPI Server
   Run the following command to start the server:

```
uvicorn app.main:app --reload
```

2. Access the API
   - Root Endpoint: http://127.0.0.1:8000/

## Making API Requests

1. /recommend Endpoint

   Example Request Body

   ```
   {
   "user_id": 1,
   "skill_id": 101
   }
   ```

   Example Response

   ```
   {
   "user_id": 1,
   "skill_id": 101,
   "predicted_rating": 4.5
   }
   ```

2. /qna

   Example Request Body

   ```
   {
   "question": "What are the best skills to learn for AI development?"
   }
   ```

   Example Response

   ```
   {
   "question": "What are the best skills to learn for AI development?",
   "answer": "The best skills to learn for AI development include Python, machine learning, and deep learning frameworks like TensorFlow or PyTorch."
   }
   ```

3. /personalized-learning-plan

   Example Request Body

   ```
   {
   "user_id": 1,
   "goal": "Become proficient in AI development"
   }
   ```

   Example Response

   ```
   {
   "user_id": 1,
   "recommended_skills": ["Skill 101", "Skill 102"],
   "learning_plan": "Week 1: Learn the basics of Skill 101. Week 2: Practice Skill 102 in real-world applications. ..."
   }
   ```

## Features Description

### Core Features

#### Recommendation System:

- Built with PyTorch.
- Provides skill recommendations for a user based on their preferences.

#### AI-Powered Features:

- Integrated OpenAI GPT for answering questions and generating personalized learning plans.

### Combined Features

#### End-to-End Learning Plans:

- Combines recommendations with GPT to provide a tailored learning experience for users.

## **Files Description**

- **`recommendation_model.py`**: Script to train the recommendation model.
- **`app/main.py`**: FastAPI application and API endpoints.
- **`app/model.py`**: Defines the recommendation model architecture.
- **`models/recommendation_model.pth`**: Trained model file.
- **`requirements.txt`**: Contains the project dependencies.

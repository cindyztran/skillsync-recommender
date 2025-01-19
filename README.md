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

## Setup

1. Clone the Repository

```
git clone https://github.com/cindyztran/skillsync-pytorch.git
cd skillsync-pytorch
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

Request Body

```
{
"user_id": 1,
"skill_id": 101
}
```

Example cURL Command

```
curl -X POST "http://127.0.0.1:8000/recommend" \
-H "Content-Type: application/json" \
-d '{"user_id": 1, "skill_id": 101}'
```

Example Response

```
{
"user_id": 1,
"skill_id": 101,
"predicted_rating": 4.5
}
```

## **Files Description**

- **`recommendation_model.py`**: Script to train the recommendation model.
- **`app/main.py`**: FastAPI application and API endpoints.
- **`app/model.py`**: Defines the recommendation model architecture.
- **`models/recommendation_model.pth`**: Trained model file.
- **`requirements.txt`**: Contains the project dependencies.

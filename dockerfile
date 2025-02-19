# Use an official Python image as the base
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (for caching efficiency)
COPY recommendation_model/requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire recommendation_model folder into /app
COPY recommendation_model /app/recommendation_model

# Ensure models directory exists
RUN mkdir -p /app/recommendation_model/models  

# Set PYTHONPATH to include /app/recommendation_model
ENV PYTHONPATH=/app/recommendation_model

# Expose the FastAPI default port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "recommendation_model.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
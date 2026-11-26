
# Use Python 3.12 lightweight image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependencies first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and trained model
COPY app/ ./app/
COPY models/model.joblib ./models/model.joblib

# Container listens on port 8000
EXPOSE 8000

# Start FastAPI using Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

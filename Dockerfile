# Base image - slim variant keeps the image small (~150 MB base)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies without caching pip files
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY app.py .
COPY models/ ./models/

# Document the port the app listens on
EXPOSE 8000

# Start the FastAPI server - bind to 0.0.0.0 so it is reachable from outside
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

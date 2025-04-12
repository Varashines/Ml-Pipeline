# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY Pipeline.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=https://dagshub.com/varaprasad7654321/Test.mlflow
ENV DAGSHUB_USERNAME=varaprasad7654321
ENV DAGSHUB_TOKEN=your_token_here
ENV PREFECT_API_URL=http://localhost:4200/api
ENV PREFECT_HOME=/app/.prefect

# Create Prefect home directory
RUN mkdir -p /app/.prefect

# Command to run the pipeline
CMD ["python", "Pipeline.py"]
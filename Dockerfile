# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# COPY requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# COPY app files
COPY api/ api/

# Expose FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
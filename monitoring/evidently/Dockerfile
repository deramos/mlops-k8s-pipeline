FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY monitor.py .

ENV PYTHONUNBUFFERED=1
ENV DRIFT_THRESHOLD=0.1
ENV PERFORMANCE_THRESHOLD=0.95

EXPOSE 5000

CMD ["uvicorn", "monitor:app", "--host", "0.0.0.0", "--port", "5000"]
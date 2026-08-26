FROM python:3.11-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src

# Mount a real trained artifact at runtime, e.g.:
#   docker run -v ./models:/app/models -e MODEL_PATH=/app/models/als_v1.pkl ...
# Training (train_baseline.py / train_embedding.py) is not run inside this image —
# it needs implicit/torch/faiss, deliberately excluded here to keep the serving
# image small and reproducible (see requirements-serving.txt).
ENV MODEL_PATH=/app/models/als_v1.pkl

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

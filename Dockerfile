FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "huggingface_hub[cli]"

RUN hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./all-MiniLM-L6-v2

CMD ["python", "agent.py"]

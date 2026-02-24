FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torch==2.7.1

EXPOSE 8000
CMD ["python", "app.py"]
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN python -m pip install --upgrade pip && \
    (test -f requirements.txt && pip install -r requirements.txt || true)
EXPOSE 8000
CMD ["python", "app.py"]
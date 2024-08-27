FROM python:3.10.12

WORKDIR /app

COPY api.py RAG_for_app.py requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt
EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
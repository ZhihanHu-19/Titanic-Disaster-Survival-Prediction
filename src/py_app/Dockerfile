FROM python:3.11-slim

WORKDIR /app

COPY src/py_app/ .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "py_app.py"]
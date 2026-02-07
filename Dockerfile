FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model at build time
RUN python model.py

EXPOSE 80

# Production WSGI server
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:80", "app:app"]


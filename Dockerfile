FROM python:3.11-slim

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app/

RUN mkdir -p /app/temp_images && chmod 777 /app/temp_images

EXPOSE 8000
CMD ["gunicorn", "web.app:app", "--bind", "0.0.0.0:8000"]

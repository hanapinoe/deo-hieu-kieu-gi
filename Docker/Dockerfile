# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11-slim

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /web
COPY . /web/

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["uvicorn", "flask_api:web", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . app/

ENV PYTHONPATH=/code

CMD ["fastapi", "run", "app/main.py", "--port", "3333"]

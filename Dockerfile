FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     libxrender1     libxext6     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip     && pip install --no-cache-dir -r requirements.txt

COPY server.py step_analyzer.py ./

ENV PORT=10000
EXPOSE 10000

CMD ["python", "server.py"]

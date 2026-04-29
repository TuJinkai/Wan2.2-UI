FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build packaging wget curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir flash-attn --no-build-isolation

COPY . .

RUN mkdir -p /app/output

EXPOSE 17861

CMD ["python", "app.py"]

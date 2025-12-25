FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.prod.txt /app/requirements.prod.txt

# 1) Install CPU-only PyTorch FIRST (prevents nvidia-*cu12 downloads)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# 2) Install your app deps
RUN pip install --no-cache-dir -r requirements.prod.txt

# 3) Install ultralytics WITHOUT deps so it doesn't pull CUDA torch/opencv-python
RUN pip install --no-cache-dir ultralytics==8.2.80 --no-deps \
 && pip install --no-cache-dir \
      matplotlib==3.9.2 scipy==1.14.1 pandas==2.2.2 seaborn==0.13.2 psutil==6.0.0 py-cpuinfo==9.0.0 ultralytics-thop==2.0.5

COPY . /app
RUN mkdir -p /app/static/uploads/m1 /app/static/uploads/m2

EXPOSE 8080
CMD ["python", "server.py"]
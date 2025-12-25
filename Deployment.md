# Docker Integration
## 1. Create requirements.prod.txt
```
Flask==3.0.3
waitress==3.0.0
opencv-python-headless==4.10.0.84
numpy==1.26.4
Pillow==10.4.0
PyYAML==6.0.2
requests==2.32.3
tqdm==4.66.5
Werkzeug==3.0.4
# ultralytics installed separately
```

## 2. Add a .dockerignore
```
__pycache__/
*.pyc
.git/
static/uploads/
test_dataset/
```

## 3. Create a Dockerfile (project root)
```
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
```

## 4. Build and run locally
```
docker build -t seaice-app .
docker run --rm -p 8080:8080 seaice-app
```

# AWS EC2 Configurations
### 1. EC2 Launch
### 2. Name instance
### 3. Choose AIM (Operating System)
`Ubuntu Server 22.04 LTS (64-bit x86)`
### 4. Choose Instance Type
`t2.micro or t3.micro`
### 5. Create a Key Pair (VERY IMPORTANT)[Need this later to connect to EC2 instance]
- Click Create new key pair
- Name: `sea-ice-key`
- Type: `RSA`
- Format: `.pem`
- Download it and keep it safe
### 6. Network settings (Security Group)
Allow the following:
- ✅ SSH — port 22 — source: My IP
- ✅ HTTP — port 80 — source: 0.0.0.0/0
- (Optional) ✅ Custom TCP 8080 — source 0.0.0.0/0
### 7. Storage
Keep the default
### 8. Launch

# Connecting to EC2
### 1. Copy public IPv4 address
`18.xxx.xxx.xxx`
### 2. SSH from your local machine
In the folder where sea-ice-key.pem is: 
```
chmod 400 sea-ice-key.pem
ssh -i sea-ice-key.pem ubuntu@<PUBLIC_IP>
```
Ex: `ssh -i sea-ice-key.pem ubuntu@18.xxx.xxx.xxx`
### 3. Once you connected install the following:
```
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```
Allow docker without sudo:
```
sudo usermod -aG docker ubuntu
exit
```
Reconnect and verify:
`docker --version`

# Deploying your application
### 1. Clone the git repo into EC2:
```
git clone <YOUR_GIT_REPO_URL>
cd python-sea-ice-classification-yolov8
```
### 2. Build the docker image
```
docker build -t seaice-app .
```
### 3. Run the docker app
```
docker run -d \
  --name seaice \
  -p 80:8080 \
  --restart unless-stopped \
  seaice-app
```

# Acess your app
### Browser:
`http://<PUBLIC_IP>/`

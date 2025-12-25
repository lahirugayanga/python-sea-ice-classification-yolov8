FROM python:3.11-slim

# Needed for opencv-python-headless in many Linux envs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for caching
COPY requirements.prod.txt /app/requirements.prod.txt
RUN pip install --no-cache-dir -r requirements.prod.txt

# Copy the rest
COPY . /app

# Ensure upload dirs exist
RUN mkdir -p /app/static/uploads/m1 /app/static/uploads/m2

EXPOSE 8080
CMD ["python", "server.py"]
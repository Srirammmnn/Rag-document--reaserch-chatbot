FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install uv for blazingly fast Python package installation
RUN pip install uv==0.4.9

# Copy and install dependencies
COPY requirements.txt .

# 1. Force CPU-only PyTorch explicitly to avoid massive GPU wheel downloads
RUN uv pip install --system --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the requirements. uv will see torch is already installed and use it.
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

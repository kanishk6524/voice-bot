FROM python:3.10-slim

# Install system dependencies (for PyAudio / PortAudio)
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Train the model at build time so it's ready at startup
RUN python train_model.py

EXPOSE 7860

ENV PORT=7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]

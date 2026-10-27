FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mantiene el contenedor encendido; el código llega por el volumen del compose
CMD ["tail", "-f", "/dev/null"]
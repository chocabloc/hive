FROM python:3.10-slim

# --- Environment Variables ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- System Dependencies ---
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# --- Application Code ---

# 1. We set the WORKDIR to a neutral location
WORKDIR /code

# 2. We copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. We copy your 'app' folder *into* /code
# This creates the /code/app structure
COPY ./app /code/app
# ***********************

# --- Run Command ---
EXPOSE 8000
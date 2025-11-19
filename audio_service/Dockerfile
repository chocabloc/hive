FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# --- Environment Variables ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# --- System Dependencies ---
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get install -y \
        python3.10 \
        python3.10-distutils \
        python3-pip \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

# --- Application Code ---

# 1. We set the WORKDIR to a neutral location
WORKDIR /code

# 2. We copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. We copy your 'app' folder *into* /code

COPY ./app /code/app


# --- Run Command ---
EXPOSE 8000
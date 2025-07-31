# Use an official Python image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | bash

# Add uv to PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Copy the project files into the container
COPY . /app


RUN pip install uv -i https://mirrors.aliyun.com/pypi/simple/


RUN uv pip install --system -e . -i https://mirrors.aliyun.com/pypi/simple/




# Expose the project directory as a volume
VOLUME ["/app"]


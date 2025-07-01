# Use an official Python image as the base
FROM rayproject/ray:latest-py311-gpu

# Set the working directory inside the container
WORKDIR /app

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . /app

# Use Aliyun mirror for setuptools, wheel, and editable install
RUN pip install --upgrade setuptools wheel -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
RUN pip install --root-user-action=ignore -e . -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# Expose the project directory as a volume
VOLUME ["/app"]

# Set the entry point to the openevolve-run.py script
ENTRYPOINT ["python", "/app/openevolve-run.py"]


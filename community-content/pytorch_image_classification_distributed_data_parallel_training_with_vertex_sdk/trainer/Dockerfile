# Stage 1: Build Environment
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime AS builder

# Install necessary tools and dependencies
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install -y google-cloud-sdk

# Copy application code
COPY . /trainer

# Set working directory
WORKDIR /trainer

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Environment
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install Google Cloud SDK
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && \
    apt-get install -y google-cloud-sdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy from the builder stage
COPY --from=builder /trainer /trainer

# Set working directory
WORKDIR /trainer

# Set the entry point
ENTRYPOINT ["python", "-m", "task"]

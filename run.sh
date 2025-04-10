#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t speech-app .

# Run the container
echo "Running container..."
docker run -p 8000:8000 speech-app 
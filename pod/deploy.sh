#!/bin/bash

# Deployment script for Video Processing Pod
# Usage: ./deploy.sh [docker-username]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Video Processing Pod Deployment${NC}"
echo -e "${GREEN}================================${NC}\n"

# Get Docker username
if [ -z "$1" ]; then
    echo -e "${YELLOW}Enter your Docker Hub username:${NC}"
    read DOCKER_USERNAME
else
    DOCKER_USERNAME=$1
fi

IMAGE_NAME="video-processor"
TAG="latest"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo -e "\n${GREEN}Step 1: Building Docker image${NC}"
echo -e "Image: ${FULL_IMAGE}\n"

docker build -t ${IMAGE_NAME}:${TAG} .

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}Build successful!${NC}"

echo -e "\n${GREEN}Step 2: Tagging image${NC}"
docker tag ${IMAGE_NAME}:${TAG} ${FULL_IMAGE}

echo -e "\n${GREEN}Step 3: Pushing to Docker Hub${NC}"
echo -e "${YELLOW}Make sure you're logged in (docker login)${NC}\n"

docker push ${FULL_IMAGE}

if [ $? -ne 0 ]; then
    echo -e "${RED}Push failed! Did you run 'docker login'?${NC}"
    exit 1
fi

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Deployment Image Ready!${NC}"
echo -e "${GREEN}================================${NC}\n"

echo -e "Docker Image: ${GREEN}${FULL_IMAGE}${NC}"
echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Go to https://www.runpod.io/console/pods"
echo -e "2. Click 'Deploy'"
echo -e "3. Select GPU (RTX 4090 recommended)"
echo -e "4. Container Settings:"
echo -e "   - Docker Image: ${GREEN}${FULL_IMAGE}${NC}"
echo -e "   - Expose HTTP Ports: ${GREEN}8000${NC}"
echo -e "   - Container Disk: ${GREEN}50GB${NC}"
echo -e "5. Deploy and save your pod URL"
echo -e "\n${YELLOW}Test your pod:${NC}"
echo -e "curl https://your-pod-id-8000.proxy.runpod.net/health"
echo -e "\n"

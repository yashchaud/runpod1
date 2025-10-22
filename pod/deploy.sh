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

echo -e "${YELLOW}💡 Tip: GitHub Actions builds are faster and don't require local disk space!${NC}"
echo -e "${YELLOW}   Just push your code and let GitHub build automatically.${NC}"
echo -e "${YELLOW}   See DOCKER_BUILD_TIPS.md for details.${NC}\n"

echo -e "${YELLOW}Do you want to build locally? (y/N):${NC}"
read -r CONFIRM
if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}Recommended: Push to GitHub and let Actions build${NC}"
    echo -e "1. git add pod/"
    echo -e "2. git commit -m \"Update pod\""
    echo -e "3. git push origin main"
    echo -e "4. Check Actions tab on GitHub"
    echo -e "\n"
    exit 0
fi

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
    echo -e "\n${RED}Build failed!${NC}"
    echo -e "\n${YELLOW}Common issues:${NC}"
    echo -e "1. ${YELLOW}No space left on device${NC}"
    echo -e "   Fix: docker system prune -a -f"
    echo -e "   Or use GitHub Actions (recommended)"
    echo -e "\n2. ${YELLOW}Docker not running${NC}"
    echo -e "   Fix: Start Docker Desktop"
    echo -e "\n3. ${YELLOW}Network issues${NC}"
    echo -e "   Fix: Check internet connection"
    echo -e "\n${YELLOW}See DOCKER_BUILD_TIPS.md for detailed solutions${NC}\n"
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

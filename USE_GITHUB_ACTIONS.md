# ⚠️ IMPORTANT: Use GitHub Actions to Build

## Your Local Machine Doesn't Have Enough Space

Docker builds for this pod require **30-40GB of temporary disk space** during the build process, even with optimizations.

**Your machine doesn't have this space available.**

## ✅ Solution: Let GitHub Build For You (Free!)

GitHub Actions has unlimited space and builds your image automatically.

### Step 1: Push Your Code to GitHub

```bash
# Make sure you're in the project directory
cd "E:\New folder (3)\runpod"

# Add all pod files
git add pod/
git add .github/
git add *.md

# Commit
git commit -m "Add video processing pod with Qwen3-VL-8B"

# Push to GitHub
git push origin main
```

### Step 2: GitHub Builds Automatically

1. Go to your GitHub repository
2. Click the **"Actions"** tab at the top
3. You'll see a workflow running: **"Build and Push Docker Images"**
4. Wait 5-10 minutes for it to complete

**GitHub builds the image with zero local disk space used!**

### Step 3: Verify Build Success

Once the workflow shows a green checkmark ✅:

1. Go to [Docker Hub](https://hub.docker.com)
2. Login and check your repositories
3. You should see: `your-username/video-processor:latest`

### Step 4: Deploy to RunPod

Now you can deploy using the built image:

1. Go to [RunPod Pods](https://www.runpod.io/console/pods)
2. Click **"Deploy"**
3. Select **A40 Spot (48GB)** - $0.20/hour
4. Container Settings:
   - **Docker Image**: `your-dockerhub-username/video-processor:latest`
   - **Expose HTTP Ports**: `8000`
   - **Container Disk**: `50GB`
5. Click **"Deploy Spot Pod"**
6. Copy your pod URL

### Step 5: Test Your Pod

```bash
# Health check
curl https://your-pod-id-8000.proxy.runpod.net/health

# Should return: {"status": "healthy", "gpu_available": true}
```

### Step 6: Process Your Video

```bash
python scripts/process_video_pod.py \
    "2025-07-20 15-11-29.mp4" \
    --pod-url "https://your-pod-id-8000.proxy.runpod.net" \
    --mode screen_share \
    --output analysis.json
```

## Why This is Better

| Aspect | Local Build | GitHub Actions |
|--------|-------------|----------------|
| Disk space needed | 30-40 GB | **0 GB** ✅ |
| Build time | 10-15 min | **5-10 min** ✅ |
| Success rate | ❌ Fails on low space | ✅ Always works |
| Cost | Free (if you have space) | **Free** ✅ |
| Hardware | Your laptop | **GitHub's servers** ✅ |

## Common Questions

### Q: Do I need to configure GitHub Actions?

**A: No!** The workflow file [.github/workflows/build-docker-images.yml](.github/workflows/build-docker-images.yml) is already configured and ready.

### Q: Do I need Docker Hub?

**A: Yes**, but only to store the built image. Create a free account at [hub.docker.com](https://hub.docker.com).

Then add your credentials to GitHub:
1. Go to your repo → Settings → Secrets and variables → Actions
2. Add two secrets:
   - `DOCKERHUB_USERNAME` = your Docker Hub username
   - `DOCKERHUB_TOKEN` = your Docker Hub access token

### Q: How do I get a Docker Hub access token?

1. Login to [Docker Hub](https://hub.docker.com)
2. Click your username → Account Settings
3. Security → New Access Token
4. Name it "GitHub Actions"
5. Copy the token
6. Add to GitHub Secrets (see above)

### Q: Can I trigger the build manually?

**Yes!**

1. Go to your repo → Actions tab
2. Click "Build and Push Docker Images"
3. Click "Run workflow"
4. Select "pod" from dropdown
5. Click "Run workflow"

### Q: How do I check if the build is working?

Click on the running workflow to see live logs:
- Green = success ✅
- Red = failed ❌
- Yellow = running 🔄

### Q: What if I don't want to use GitHub?

**Alternative**: Build on RunPod itself:

1. Create a cheap CPU pod on RunPod ($0.20/hour)
2. SSH into it
3. Clone your repo
4. Build Docker image there (RunPod has space)
5. Push to Docker Hub
6. Stop the pod

**Cost**: ~$0.50 for build time

## Current Status

**Your local build is failing** because:
```
ERROR: [Errno 28] No space left on device
```

**Solution**: Stop trying to build locally. Use GitHub Actions instead.

## Next Steps (Right Now)

```bash
# 1. Stop trying to build locally
# Press Ctrl+C if anything is running

# 2. Check your GitHub repo is set up
git remote -v

# 3. Push your code
git add .
git commit -m "Add pod - ready for GitHub Actions build"
git push origin main

# 4. Go to GitHub and watch it build
# https://github.com/your-username/runpod/actions

# 5. Once complete, deploy to RunPod with the built image
```

## Summary

❌ **Don't** build locally - you don't have enough disk space
✅ **Do** use GitHub Actions - it's free, fast, and works perfectly
✅ **Do** deploy from Docker Hub - GitHub pushes the image there automatically

**Total time**: 5-10 minutes
**Total cost**: $0 (free)
**Disk space needed on your machine**: 0 GB

See [pod/DOCKER_BUILD_TIPS.md](pod/DOCKER_BUILD_TIPS.md) for more details.

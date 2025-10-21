# Video Understanding Agent - RunPod Serverless

A real-time video understanding system using RunPod serverless endpoints for scalable, cost-effective AI inference.

## 🎯 Project Status

**Current Phase**: Phase 1 Complete ✓

- ✅ Phase 0: 40% (Automated tasks complete, manual RunPod setup pending)
- ✅ Phase 1: 100% Complete
- ⏳ Phase 2-12: Not started

See [PROGRESS.md](PROGRESS.md) for detailed status.

---

## 🚀 Quick Start

### Prerequisites Completed ✓

- ✅ Python 3.13.3 installed
- ✅ Docker 28.3.0 installed
- ✅ Git 2.49.0 installed
- ✅ Project structure created
- ✅ Virtual environment configured
- ✅ Dependencies installed

### Next Steps (You Need To Do)

1. **Create RunPod Account** ⏳
   - Go to https://www.runpod.io/
   - Sign up and verify email
   - Add credits ($25-50 recommended for testing)

2. **Get API Key** ⏳
   - Login to RunPod dashboard
   - Settings → API Keys → Create API Key
   - Copy the key

3. **Configure Environment** ⏳
   ```powershell
   # Edit .env file and add your API key
   notepad .env
   # Replace: RUNPOD_API_KEY=your_runpod_api_key_here
   ```

4. **Verify Setup**
   ```powershell
   .\venv\Scripts\python.exe scripts\test_setup.py
   ```

---

## 📁 Project Structure

```
e:\New folder (3)\runpod\
├── config/              # Configuration files
├── services/            # Service implementations
│   ├── ingestion/      # Video/audio ingestion
│   ├── detection/      # Object detection (RunPod YOLO client)
│   ├── audio/          # Audio processing (RunPod Whisper client)
│   └── vlm/            # Vision-language model (RunPod VLM client)
├── data/               # Data storage
│   ├── videos/         # Input videos
│   └── frames/         # Extracted frames
├── scripts/            # Utility scripts
├── utils/              # Helper utilities
├── tests/              # Test files
├── venv/               # Python virtual environment
├── .env                # Environment variables (ADD YOUR API KEY HERE!)
├── requirements.txt    # Python dependencies
└── IMPLEMENTATION_PLAN.md  # Detailed implementation guide
```

---

## 🔧 Architecture

This system uses **RunPod Serverless Endpoints** instead of local GPU inference:

```
LiveKit Stream → Ingestion Service → Redis
                                      ↓
                           ┌──────────┴──────────┐
                           ↓                     ↓
                 Detection Service      Audio Service
                 (calls RunPod          (calls RunPod
                  YOLO endpoint)        Whisper endpoint)
                           ↓                     ↓
                           └──────────┬──────────┘
                                      ↓
                              VLM Reasoning Service
                              (calls RunPod VLM endpoint)
                                      ↓
                              PostgreSQL + Events
```

### Why RunPod Serverless?

✅ **No expensive GPUs needed** - No RTX 4090, A100, etc.
✅ **Auto-scaling** - Scales from 0 to N workers automatically
✅ **Pay-per-second** - Only pay when processing
✅ **Develop anywhere** - Works on any laptop (Mac, Windows, Linux)
✅ **Cost-effective** - ~$30-50/hour vs $200-400 for dedicated GPU

---

## 📋 Implementation Guide

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the complete, phase-by-phase implementation guide with 200+ testable tasks.

### Phase Overview

1. **Phase 0-1**: Environment Setup ✅ (mostly complete)
2. **Phase 2**: Database & Storage Setup
3. **Phase 3**: RunPod Endpoint Deployment
4. **Phase 4**: Ingestion Service
5. **Phase 5**: Detection Service (RunPod client)
6. **Phase 6**: Audio Processing Service (RunPod client)
7. **Phase 7**: VLM Reasoning Service (RunPod client)
8. **Phase 8**: Docker Orchestration
9. **Phase 9**: End-to-End Testing
10. **Phase 10-12**: Advanced Features & Production

---

## 🛠️ Development

### Activate Virtual Environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Or directly run Python
.\venv\Scripts\python.exe
```

### Install Dependencies

```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

### Run Tests

```powershell
.\venv\Scripts\python.exe scripts\test_setup.py
```

### Building Docker Images

**Option 1: Automated with GitHub Actions (Recommended)**

Push your code to GitHub and let GitHub Actions build Docker images automatically:

```bash
git add .
git commit -m "Update endpoint code"
git push origin main
```

Images are built 3x faster than local builds and pushed to Docker Hub automatically. See [GITHUB_ACTIONS_SETUP.md](GITHUB_ACTIONS_SETUP.md) for setup instructions.

**Option 2: Manual Local Build**

```bash
# Build YOLO endpoint
docker build -t <your-dockerhub-username>/yolo11-runpod:latest ./endpoints/yolo
docker push <your-dockerhub-username>/yolo11-runpod:latest

# Build Whisper endpoint
docker build -t <your-dockerhub-username>/whisper-runpod:latest ./endpoints/whisper
docker push <your-dockerhub-username>/whisper-runpod:latest
```

---

## 📚 Documentation

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Complete implementation guide
- [PRD.md](PRD.md) - Product requirements document
- [PROGRESS.md](PROGRESS.md) - Current progress tracker
- [PHASE_0_1_SETUP_GUIDE.md](PHASE_0_1_SETUP_GUIDE.md) - Setup instructions
- [PHASE_0_1_COMPLETE.md](PHASE_0_1_COMPLETE.md) - Completion summary
- [GITHUB_ACTIONS_SETUP.md](GITHUB_ACTIONS_SETUP.md) - Automated Docker builds with GitHub Actions
- [RUNPOD_ENDPOINT_DEPLOYMENT_GUIDE.md](RUNPOD_ENDPOINT_DEPLOYMENT_GUIDE.md) - RunPod deployment guide
- [ENDPOINT_RESEARCH_SUMMARY.md](ENDPOINT_RESEARCH_SUMMARY.md) - Endpoint research findings

---

## 🔑 Environment Variables

Required in `.env` file:

```bash
# RunPod (Required for Phase 3+)
RUNPOD_API_KEY=your_key_here              # ⚠️ REQUIRED
RUNPOD_YOLO_ENDPOINT_ID=                   # From Phase 3
RUNPOD_WHISPER_ENDPOINT_ID=                # From Phase 3
RUNPOD_VLM_ENDPOINT_ID=                    # From Phase 3

# Database (Required for Phase 2+)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=videoagent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=change_me_in_production

# Redis (Required for Phase 2+)
REDIS_HOST=localhost
REDIS_PORT=6379

# Storage (Required for Phase 2+)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=changeme123
```

---

## 💰 Cost Estimation

Based on RunPod serverless pricing:

- **YOLO Detection**: ~$0.0004 per image (RTX 4090)
- **Whisper ASR**: ~$0.001 per minute of audio (A40)
- **VLM Reasoning**: ~$0.003 per inference (A100)

**Example**: Processing 1 hour of 30fps video ≈ $30-50

Compare to dedicated GPU server: $200-400/hour

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

### "Virtual environment not found"
```powershell
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt
```

### "RunPod API key not set"
Edit `.env` and add your RunPod API key from https://www.runpod.io/console

---

## 🤝 Contributing

This is an implementation of the [PRD.md](PRD.md) specification using RunPod serverless infrastructure.

---

## 📄 License

[Add your license here]

---

## 🔗 Resources

- **RunPod**: https://www.runpod.io/
- **RunPod Docs**: https://docs.runpod.io/
- **RunPod Serverless**: https://docs.runpod.io/serverless/overview

---

**Status**: ✅ Phase 1 Complete | ⏳ Awaiting RunPod Account Setup | 🚀 Ready to Deploy Endpoints

*Last updated: 2025-01-XX*

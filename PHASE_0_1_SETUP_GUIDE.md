# PHASE 0 & 1 SETUP GUIDE

## Status: In Progress

This guide will walk you through completing Phase 0 (Pre-flight Checks) and Phase 1 (Environment Setup).

---

## PHASE 0: PRE-FLIGHT CHECKS

### Task 0.1: Create RunPod Account ⏳ MANUAL ACTION REQUIRED

**Action Required:**
1. Go to https://www.runpod.io/
2. Click "Sign Up" or "Get Started"
3. Create account with email or GitHub
4. Verify your email

**Verification:**
- [ ] I can log into RunPod dashboard

---

### Task 0.2: Get RunPod API Key ⏳ MANUAL ACTION REQUIRED

**Action Required:**
1. Log into RunPod dashboard: https://www.runpod.io/console
2. Click on your profile/settings (top right)
3. Navigate to "API Keys" section
4. Click "Create API Key" or "Generate API Key"
5. Give it a name like "video-agent-dev"
6. Copy the API key (starts with something like `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX`)

**Verification:**
- [ ] I have copied my RunPod API key
- [ ] API key saved securely (we'll add to .env file later)

**Your RunPod API Key:**
```
PASTE_YOUR_API_KEY_HERE (keep this file private!)
```

---

### Task 0.3: Add Billing/Credits to RunPod ⏳ MANUAL ACTION REQUIRED

**Action Required:**
1. In RunPod dashboard, go to "Billing" or "Credits"
2. Add payment method OR buy credits
3. Recommended: Start with $25-50 for testing
4. Set up spending alerts (optional but recommended)

**Verification:**
- [ ] Credits/balance visible in RunPod dashboard
- [ ] Spending alerts configured (optional)

**Current Balance:** $_______

---

### Task 0.4: Check Available Disk Space ✅ AUTOMATED

**Command:**
```powershell
Get-PSDrive -PSProvider FileSystem | Where-Object {$_.Name -eq 'E'}
```

**Expected:** At least 500GB free space

**Status:** Will check automatically...

---

### Task 0.5: Verify Network Connectivity ✅ AUTOMATED

**Commands:**
```powershell
# Test general internet
Test-Connection -ComputerName 8.8.8.8 -Count 3 -Quiet

# Test RunPod API
Invoke-WebRequest -Uri https://api.runpod.ai -Method Head
```

**Status:** Will check automatically...

---

## PHASE 1: ENVIRONMENT SETUP

### Current System Check ✅ COMPLETED

- ✅ Python 3.13.3 installed (3.11+ required)
- ✅ Docker 28.3.0 installed
- ✅ Git 2.49.0 installed

### Task 1.1.1: Update System Packages ⏳ MANUAL

**For Windows:**
- Windows Update is recommended
- Or skip if system is recent

**Verification:**
- [ ] System is up to date

---

### Task 1.1.2-1.1.5: System Dependencies ✅ MOSTLY COMPLETE

- ✅ Docker already installed
- ✅ Python 3.13 already installed (>3.11 ✓)
- ⏳ Git already installed
- ❌ GStreamer not checked yet
- ❌ FFmpeg not checked yet

---

### Task 1.2.1: Create Project Directory Structure ✅ AUTOMATED

Will create:
```
e:\New folder (3)\runpod\
├── config/
├── models/           (not needed for RunPod, but keep for consistency)
├── services/
│   ├── ingestion/
│   ├── detection/
│   ├── audio/
│   ├── fusion/
│   ├── vlm/
│   ├── events/
│   ├── highlights/
│   └── export/
├── data/
│   ├── videos/
│   ├── frames/
│   └── events/
├── logs/
├── outputs/
├── scripts/
├── utils/
└── tests/
```

---

### Task 1.2.2-1.2.3: Git Setup ✅ AUTOMATED

Will initialize git repository and create .gitignore

---

### Task 1.3.1-1.3.4: Python Environment ✅ AUTOMATED

Will:
1. Create Python virtual environment
2. Upgrade pip
3. Install RunPod SDK
4. Install basic CV libraries

---

## AUTOMATED SETUP SCRIPT

I've created an automated script to complete what can be automated.

**What it does:**
- ✅ Checks disk space
- ✅ Tests network connectivity to RunPod
- ✅ Creates project directory structure
- ✅ Initializes Git repository
- ✅ Creates .gitignore
- ✅ Creates Python virtual environment
- ✅ Installs required Python packages
- ✅ Creates .env template file

**What you need to do manually:**
1. Create RunPod account
2. Get RunPod API key
3. Add credits to RunPod
4. Fill in API key in .env file

---

## NEXT STEPS

After completing Phase 0 & 1:
1. Add your RunPod API key to `.env` file
2. Proceed to Phase 2: Database & Storage Setup
3. Or jump to Phase 3: Deploy RunPod endpoints

---

## VERIFICATION CHECKLIST

### Phase 0 ✓
- [ ] RunPod account created
- [ ] RunPod API key obtained
- [ ] Credits added to RunPod account
- [ ] Disk space > 500GB
- [ ] Network connectivity to RunPod API confirmed

### Phase 1 ✓
- [ ] Project directory structure created
- [ ] Git repository initialized
- [ ] .gitignore created
- [ ] Python virtual environment created
- [ ] RunPod SDK installed
- [ ] Basic dependencies installed
- [ ] .env template created

---

**Run the automated setup script:** `setup_phase_0_1.ps1`

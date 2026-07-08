# 📦 Installation Guide

Complete installation instructions for DataWhisperer on all supported platforms.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Docker Installation (Recommended)](#docker-installation-recommended)
- [Native Installation](#native-installation)
  - [Windows](#windows)
  - [Linux](#linux)
  - [macOS](#macos)
- [Ollama Setup](#ollama-setup)
- [Post-Installation Verification](#post-installation-verification)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 10 GB free | 20 GB free |
| **Python** | 3.11 | 3.12 |
| **OS** | Windows 10+, Ubuntu 20.04+, macOS 12+ | Latest stable |
| **GPU** (optional) | NVIDIA with 6 GB VRAM | NVIDIA with 8+ GB VRAM |

> [!NOTE]
> The AI model (`qwen2.5:7b`) requires ~4.7 GB disk and ~5 GB RAM during inference.
> Smaller models like `qwen2.5:3b` work with less RAM.

---

## Docker Installation (Recommended)

Docker provides the most consistent experience across platforms. Everything runs with one command.

### 1. Install Docker

**Windows:**
1. Download [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. Run the installer (requires WSL 2 — installer will guide you)
3. Restart your computer
4. Open Docker Desktop and wait for it to start

**Linux (Ubuntu/Debian):**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose plugin
sudo apt-get install docker-compose-plugin
```

**macOS:**
1. Download [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
   - Apple Silicon (M1/M2/M3): Use the **Apple chip** version
   - Intel: Use the **Intel chip** version
2. Drag to Applications and launch

### 2. Verify Docker

```bash
docker --version          # Should show 24.0+
docker compose version    # Should show v2.20+
```

### 3. Launch DataWhisperer

```bash
# Clone the repository
git clone https://github.com/your-org/datawhisperer.git
cd datawhisperer

# Create environment config
cp .env.example .env

# Start everything
docker compose up -d
```

Open **http://localhost:8501** in your browser.

### 4. Follow Startup Progress

```bash
# Watch all logs
docker compose logs -f

# Check service health
docker compose ps
```

> [!IMPORTANT]
> First launch downloads the AI model (~4 GB). This is a one-time download.
> Subsequent starts take ~10 seconds.

---

## Native Installation

### Windows

#### Prerequisites
1. **Python 3.11 or 3.12** — Download from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation
2. **Git** — Download from [git-scm.com](https://git-scm.com/download/win)
3. **Ollama** — Download from [ollama.com](https://ollama.com/download/windows)

#### Steps

```powershell
# 1. Verify prerequisites
python --version    # Should show 3.11.x or 3.12.x
ollama --version    # Should show 0.1.x or higher

# 2. Pull the AI model
ollama pull qwen2.5:7b

# 3. Clone the project
git clone https://github.com/your-org/datawhisperer.git
cd datawhisperer

# 4. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 5. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 6. Configure
copy .env.example .env
# Edit .env with notepad if needed: notepad .env

# 7. Start Ollama (if not already running)
# Ollama runs as a system service on Windows — check the system tray icon

# 8. Run DataWhisperer
streamlit run app.py
```

> [!TIP]
> On Windows, Ollama starts automatically as a background service after installation.
> Check the system tray for the Ollama icon.

### Linux

#### Prerequisites

```bash
# Python 3.11+ (Ubuntu 22.04+ has it, or use deadsnakes PPA)
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip git curl

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

#### Steps

```bash
# 1. Pull the AI model
ollama pull qwen2.5:7b

# 2. Clone and setup
git clone https://github.com/your-org/datawhisperer.git
cd datawhisperer

# 3. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Configure
cp .env.example .env

# 6. Run
streamlit run app.py
```

#### Running as a Systemd Service (Production)

```bash
sudo tee /etc/systemd/system/datawhisperer.service > /dev/null << 'EOF'
[Unit]
Description=DataWhisperer AI Data Analyst
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/datawhisperer
Environment="PATH=/opt/datawhisperer/.venv/bin:/usr/bin"
ExecStart=/opt/datawhisperer/.venv/bin/streamlit run app.py \
    --server.headless=true \
    --server.address=0.0.0.0 \
    --server.port=8501
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now datawhisperer
sudo systemctl status datawhisperer
```

### macOS

#### Prerequisites

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.12
brew install python@3.12

# Install Ollama
brew install ollama
```

#### Steps

```bash
# 1. Start Ollama and pull model
ollama serve &    # Or: brew services start ollama
ollama pull qwen2.5:7b

# 2. Clone and setup
git clone https://github.com/your-org/datawhisperer.git
cd datawhisperer

# 3. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Configure
cp .env.example .env

# 6. Run
streamlit run app.py
```

> [!NOTE]
> On Apple Silicon (M1/M2/M3), Ollama runs natively with excellent performance.
> GPU acceleration is automatic via Metal — no extra configuration needed.

---

## Ollama Setup

### Choosing a Model

| Model | Size | RAM Required | Best For |
|-------|------|-------------|----------|
| `qwen2.5:3b` | 2.0 GB | 4 GB | Low-resource machines |
| `qwen2.5:7b` | 4.7 GB | 8 GB | **Recommended default** |
| `qwen2.5:14b` | 9.0 GB | 16 GB | Higher accuracy |
| `codellama:7b` | 3.8 GB | 8 GB | Code-focused alternative |
| `llama3.1:8b` | 4.7 GB | 8 GB | General purpose alternative |

### Change the Model

```bash
# Pull a different model
ollama pull qwen2.5:3b

# Update .env
# OLLAMA_MODEL=qwen2.5:3b

# Or for Docker, set before starting:
OLLAMA_MODEL=qwen2.5:3b docker compose up -d
```

### Verify Ollama is Working

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Test inference
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b",
  "prompt": "Write a Python function to add two numbers",
  "stream": false
}'
```

---

## Post-Installation Verification

Run these checks after installation:

```bash
# 1. Check Ollama
curl -s http://localhost:11434/api/tags | python -m json.tool

# 2. Check DataWhisperer (should return "ok")
curl -s http://localhost:8501/_stcore/health

# 3. For Docker installations
docker compose ps    # All services should show "healthy"
```

### Expected Output

```
NAME                        STATUS              PORTS
datawhisperer-app           Up (healthy)        0.0.0.0:8501->8501/tcp
datawhisperer-ollama        Up (healthy)        0.0.0.0:11434->11434/tcp
```

---

## Next Steps

- 📖 [Deployment Guide](DEPLOY.md) — Production deployment strategies
- 🔧 [Troubleshooting](TROUBLESHOOTING.md) — Common issues and fixes
- ⚡ [Optimization Guide](OPTIMIZATION.md) — Performance tuning

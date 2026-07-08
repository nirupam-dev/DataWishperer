# ⚡ Optimization Guide

Performance tuning for DataWhisperer — GPU acceleration, CPU optimization, and memory management.

---

## Table of Contents

- [GPU Support (NVIDIA)](#gpu-support-nvidia)
- [GPU Support (Apple Silicon)](#gpu-support-apple-silicon)
- [CPU Optimization](#cpu-optimization)
- [Memory Optimization](#memory-optimization)
- [Model Selection](#model-selection)
- [Docker Resource Tuning](#docker-resource-tuning)
- [Platform-Specific Tuning](#platform-specific-tuning)

---

## GPU Support (NVIDIA)

GPU acceleration provides **5–10x faster** LLM inference.

### Prerequisites

1. **NVIDIA GPU** with 6+ GB VRAM (RTX 3060 or better recommended)
2. **NVIDIA Driver** 525+ installed
3. **NVIDIA Container Toolkit** (for Docker GPU passthrough)

### Install NVIDIA Container Toolkit (Linux)

```bash
# Add the NVIDIA repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Install NVIDIA Container Toolkit (Windows)

GPU support in Docker on Windows requires WSL 2:

1. Ensure **WSL 2** is enabled and running a Linux distro
2. Install the latest **NVIDIA Windows driver** (Game Ready or Studio)
3. Docker Desktop → Settings → Resources → WSL Integration → Enable for your distro
4. GPU passthrough is automatic with modern Docker Desktop on WSL 2

### Launch with GPU

```bash
# Docker (recommended)
docker compose --profile gpu up -d

# Or with Make
make docker-up-gpu

# Verify GPU is being used
docker exec datawhisperer-ollama-gpu nvidia-smi
```

### Native Ollama with GPU

```bash
# Ollama auto-detects NVIDIA GPUs. Verify:
ollama ps
# Should show GPU layers loaded

# Check GPU utilization during inference
nvidia-smi -l 1   # Updates every second
```

---

## GPU Support (Apple Silicon)

Apple M1/M2/M3/M4 chips provide GPU acceleration via **Metal** — no extra setup required.

### Native (Recommended for Mac)

```bash
# Install Ollama natively (not in Docker)
brew install ollama
ollama serve &

# Pull and run — Metal acceleration is automatic
ollama pull qwen2.5:7b
```

### Verify Metal Acceleration

```bash
# During inference, check Activity Monitor:
# → GPU History should show activity
# Or check Ollama logs:
ollama ps
# Should show "metal" in the acceleration column
```

> [!TIP]
> For best performance on Mac, run Ollama **natively** (not in Docker).
> Docker on Mac uses a Linux VM, which cannot access Metal GPU.

### Mac-Specific .env Tuning

```bash
# Apple M-series can handle larger contexts efficiently
OLLAMA_NUM_CTX=8192
OLLAMA_NUM_PREDICT=4096
```

---

## CPU Optimization

For systems without a GPU, these settings maximize CPU inference performance.

### Ollama CPU Settings

```bash
# Set thread count to match physical cores (not logical)
# Example: 8-core CPU → 8 threads
export OLLAMA_NUM_THREADS=8

# For Docker, add to docker-compose.yml under ollama service:
# environment:
#   - OLLAMA_NUM_THREADS=8
```

### Use a Quantized Model

Smaller, quantized models run significantly faster on CPU:

```bash
# 3B model — fastest CPU inference
ollama pull qwen2.5:3b
# Update .env: OLLAMA_MODEL=qwen2.5:3b

# Q4 quantized — smaller memory footprint
ollama pull qwen2.5:7b-instruct-q4_0
```

### Reduce Context Window

```bash
# Smaller context = faster inference + less memory
OLLAMA_NUM_CTX=2048   # Down from default 4096
OLLAMA_NUM_PREDICT=1024  # Shorter responses
```

### System-Level Optimizations

**Linux:**
```bash
# Use performance CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase file descriptor limits
ulimit -n 65535
```

**Windows:**
- Set Power Plan to **High Performance** (Control Panel → Power Options)
- Close unnecessary background applications
- Disable Windows Search indexing for the project directory

---

## Memory Optimization

### Reduce Application Memory

```bash
# .env settings for constrained environments
SANDBOX_MAX_MEMORY_MB=256     # Down from 512
OLLAMA_NUM_CTX=2048           # Down from 4096
STORAGE_MAX_FILE_SIZE_MB=25   # Limit upload size
```

### Docker Memory Limits

In `docker-compose.yml`:

```yaml
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 6G   # Adjust based on available RAM

  app:
    deploy:
      resources:
        limits:
          memory: 1G   # Streamlit + pandas needs ~1 GB
```

### Memory Usage by Component

| Component | Typical Usage | Notes |
|-----------|--------------|-------|
| Ollama (7B model) | 4–6 GB | Loaded into RAM on first query |
| Streamlit + Backend | 200–500 MB | Depends on dataset size |
| Sandbox processes | 256–512 MB | Per execution, short-lived |
| **Total** | **5–7 GB** | With 7B model |
| **Total (3B model)** | **3–4 GB** | Budget-friendly option |

---

## Model Selection

Choose the right model for your hardware:

| Model | Disk | RAM (CPU) | VRAM (GPU) | Speed | Quality |
|-------|------|-----------|------------|-------|---------|
| `qwen2.5:3b` | 2.0 GB | 4 GB | 3 GB | ⚡⚡⚡⚡ | ★★★ |
| `qwen2.5:7b` | 4.7 GB | 8 GB | 6 GB | ⚡⚡⚡ | ★★★★ |
| `qwen2.5:14b` | 9.0 GB | 16 GB | 10 GB | ⚡⚡ | ★★★★★ |
| `codellama:7b` | 3.8 GB | 8 GB | 5 GB | ⚡⚡⚡ | ★★★★ |
| `deepseek-coder:6.7b` | 3.8 GB | 8 GB | 5 GB | ⚡⚡⚡ | ★★★★ |

### Switching Models

```bash
# Pull the new model
ollama pull qwen2.5:3b

# Update .env
OLLAMA_MODEL=qwen2.5:3b

# Restart
docker compose restart app
# Or for native: restart streamlit
```

---

## Docker Resource Tuning

### Docker Desktop (Windows / macOS)

1. Open Docker Desktop → **Settings** → **Resources**
2. Recommended settings:

| Setting | Minimum | Recommended |
|---------|---------|-------------|
| CPUs | 4 | 6–8 |
| Memory | 6 GB | 10–12 GB |
| Swap | 1 GB | 2 GB |
| Disk image | 20 GB | 40 GB |

### Docker Engine (Linux)

Docker on Linux uses host resources directly — no configuration needed.
Set container-level limits in `docker-compose.yml` (see above).

---

## Platform-Specific Tuning

### Windows

```powershell
# Check available GPU
nvidia-smi

# Set Ollama to use all GPU memory
$env:OLLAMA_MAX_VRAM = "0"  # 0 = use all available

# High performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

### Linux

```bash
# Use all available GPU memory
export OLLAMA_MAX_VRAM=0

# Pin Ollama to specific CPU cores (NUMA optimization)
taskset -c 0-7 ollama serve

# Increase shared memory for large datasets
echo "tmpfs /dev/shm tmpfs defaults,size=4g 0 0" | sudo tee -a /etc/fstab
```

### macOS

```bash
# Metal is auto-detected, but ensure correct binary
file $(which ollama)
# Should show: Mach-O 64-bit executable arm64 (on Apple Silicon)

# Increase file descriptor limit for large datasets
ulimit -n 10240
```

---

## Performance Benchmarks

Approximate inference times for a typical data analysis query:

| Setup | Model | First Query | Subsequent |
|-------|-------|-------------|------------|
| RTX 4090 | 7B | ~2s | ~1s |
| RTX 3060 | 7B | ~5s | ~3s |
| M2 Pro | 7B | ~4s | ~2s |
| CPU (8-core) | 7B | ~30s | ~15s |
| CPU (8-core) | 3B | ~10s | ~5s |

> First query includes model loading time. Subsequent queries are faster because the model stays in memory.

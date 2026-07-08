# 🔧 Troubleshooting Guide

Solutions for common DataWhisperer issues, organized by category.

---

## Table of Contents

- [Docker Issues](#docker-issues)
- [Ollama / LLM Issues](#ollama--llm-issues)
- [Application Issues](#application-issues)
- [Platform-Specific Issues](#platform-specific-issues)
- [Performance Issues](#performance-issues)
- [Data & Upload Issues](#data--upload-issues)
- [Diagnostic Commands](#diagnostic-commands)

---

## Docker Issues

### Container fails to start

**Symptom:** `docker compose up` exits immediately or containers keep restarting.

```bash
# Check what went wrong
docker compose logs app
docker compose logs ollama

# Check resource usage
docker stats --no-stream
```

**Common causes:**
| Cause | Fix |
|-------|-----|
| Port 8501 already in use | `APP_PORT=8502` in `.env`, or stop the conflicting process |
| Port 11434 already in use | Stop local Ollama: `ollama stop` or change `OLLAMA_HOST_PORT` |
| Insufficient memory | Increase Docker Desktop memory limit to 8 GB+ |
| Docker daemon not running | Start Docker Desktop / `sudo systemctl start docker` |

### "No space left on device"

```bash
# Clean unused Docker resources
docker system prune -a --volumes

# Check disk usage
docker system df
```

### Container shows "unhealthy"

```bash
# Check which service is unhealthy
docker compose ps

# Test health endpoints manually
docker exec datawhisperer-app curl -f http://localhost:8501/_stcore/health
docker exec datawhisperer-ollama curl -f http://localhost:11434/api/tags
```

### Build fails on ARM / Apple Silicon

If numpy/scipy wheels fail to build:

```bash
# Force platform in docker-compose.yml under the app service:
# platform: linux/amd64

# Or build with explicit platform
docker compose build --build-arg PYTHON_VERSION=3.12-slim
```

---

## Ollama / LLM Issues

### "Connection refused" to Ollama

**Symptom:** App shows "Cannot connect to Ollama" or timeout errors.

**Docker deployment:**
```bash
# Verify Ollama container is running
docker compose ps ollama

# Check Ollama logs
docker compose logs ollama

# Test from app container
docker exec datawhisperer-app curl http://ollama:11434/api/tags
```

**Native deployment:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Windows: Check system tray for Ollama icon
# macOS: brew services start ollama
```

### Model not found / not pulled

```bash
# List available models
ollama list

# Pull the required model
ollama pull qwen2.5:7b

# For Docker, restart model-init
docker compose restart model-init
docker compose logs model-init
```

### LLM responses are slow

| Cause | Solution |
|-------|----------|
| CPU inference on large model | Switch to `qwen2.5:3b` in `.env` |
| Insufficient RAM | Close other applications, add swap |
| No GPU acceleration | See [Optimization Guide](OPTIMIZATION.md) |
| Large context window | Reduce `OLLAMA_NUM_CTX` to `2048` |

### LLM generates incorrect code

```bash
# Lower temperature for more deterministic output
# In .env:
OLLAMA_TEMPERATURE=0.05

# Increase context window if data schema is large
OLLAMA_NUM_CTX=8192
```

---

## Application Issues

### Streamlit "Please wait..." forever

**Causes:**
1. Ollama not reachable — check connection (see above)
2. Model still downloading — check `docker compose logs model-init`
3. Browser cache stale — hard refresh with `Ctrl+Shift+R`

### "ModuleNotFoundError"

```bash
# Ensure virtual environment is activated
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Charts not rendering

```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"
# Should output: Agg (for headless/Docker)

# If not, set environment variable:
export MPLBACKEND=Agg  # Linux/Mac
set MPLBACKEND=Agg     # Windows
```

### Session state errors after upgrade

```bash
# Clear Streamlit cache
streamlit cache clear

# Or delete the cache directory
# Windows: del /s /q %USERPROFILE%\.streamlit\cache
# Linux:   rm -rf ~/.streamlit/cache
```

---

## Platform-Specific Issues

### Windows

| Issue | Solution |
|-------|----------|
| `python` not found | Use `python3` or reinstall with "Add to PATH" checked |
| Permission denied on venv | Run PowerShell as Administrator |
| `make` not recognized | Install via `choco install make` or use commands directly |
| WSL 2 not enabled for Docker | Enable in Windows Features or run: `wsl --install` |
| Long path errors | Enable long paths: `git config --global core.longpaths true` |

### Linux

| Issue | Solution |
|-------|----------|
| Permission denied on Docker | `sudo usermod -aG docker $USER && newgrp docker` |
| Python 3.12 not available | Use deadsnakes PPA: `sudo add-apt-repository ppa:deadsnakes/ppa` |
| `pip` installs to system | Always use a virtual environment |
| GPU not detected in Docker | Install `nvidia-container-toolkit` (see Optimization Guide) |

### macOS

| Issue | Solution |
|-------|----------|
| Homebrew Python conflicts | Use `python3.12` explicitly, not `python` |
| Rosetta translation issues | Ensure Ollama is the ARM-native version |
| Docker Desktop slow | Increase memory/CPU in Docker Desktop → Settings → Resources |
| SSL certificate errors | `pip install certifi && pip install --upgrade pip` |

---

## Performance Issues

### High memory usage

```bash
# Check per-container memory
docker stats --no-stream

# Reduce memory limits in .env:
SANDBOX_MAX_MEMORY_MB=256
OLLAMA_NUM_CTX=2048

# Use a smaller model:
OLLAMA_MODEL=qwen2.5:3b
```

### Slow CSV processing

```bash
# Check file size — large files take longer
# Reduce max columns if needed:
STORAGE_MAX_COLUMNS=200

# For very large files (>50 MB), increase limit:
STORAGE_MAX_FILE_SIZE_MB=100
```

---

## Data & Upload Issues

### "File too large"

Increase the limit in `.env`:
```
STORAGE_MAX_FILE_SIZE_MB=100
```

For Docker, also update Streamlit's limit:
```bash
# In .streamlit/config.toml, add:
# [server]
# maxUploadSize = 100
```

### CSV encoding errors

DataWhisperer auto-detects encoding, but if issues persist:
1. Open the CSV in a text editor
2. Save as UTF-8 encoding
3. Re-upload

### "Too many columns"

```bash
# Increase column limit in .env:
STORAGE_MAX_COLUMNS=1000
```

---

## Diagnostic Commands

### Quick Health Check Script

```bash
echo "=== DataWhisperer Diagnostics ==="

echo -n "Docker:    "
docker --version 2>/dev/null && echo "OK" || echo "NOT INSTALLED"

echo -n "Compose:   "
docker compose version 2>/dev/null && echo "OK" || echo "NOT INSTALLED"

echo -n "Python:    "
python3 --version 2>/dev/null || echo "NOT INSTALLED"

echo -n "Ollama:    "
curl -sf http://localhost:11434/api/tags > /dev/null && echo "RUNNING" || echo "NOT REACHABLE"

echo -n "App:       "
curl -sf http://localhost:8501/_stcore/health > /dev/null && echo "HEALTHY" || echo "NOT REACHABLE"

echo ""
echo "=== Docker Services ==="
docker compose ps 2>/dev/null || echo "No compose project running"

echo ""
echo "=== Resource Usage ==="
docker stats --no-stream 2>/dev/null || echo "No containers running"
```

### Collect Debug Info for Bug Reports

```bash
echo "--- System ---"
uname -a
echo "--- Docker ---"
docker version
docker compose version
echo "--- Containers ---"
docker compose ps
echo "--- Logs (last 50 lines) ---"
docker compose logs --tail=50 app
docker compose logs --tail=50 ollama
echo "--- Resources ---"
docker stats --no-stream
```

---

## Still Stuck?

1. Check the [GitHub Issues](https://github.com/your-org/datawhisperer/issues) for known problems
2. Create a new issue with the output of the diagnostic commands above
3. Include your OS, Docker version, and the complete error message

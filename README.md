# 🔮 DataWhisperer — Talk to Your CSV with AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Upload a CSV. Ask questions in plain English. Get tables, charts, and insights — powered by Groq (primary) with automatic Ollama local fallback. Your data stays on your machine.**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Natural Language Queries** | Ask questions about your data in plain English |
| 🔒 **Secure Sandbox Execution** | Generated code runs in process-isolated sandbox with AST validation |
| 📊 **Smart Visualization** | Automatic chart type selection with dark-themed rendering |
| 🧠 **Dual AI Providers** | Groq cloud LLM (primary) with automatic Ollama local fallback |
| 📈 **Advanced Analytics** | Data profiling, statistical analysis, predictive modeling |
| 📥 **Export** | Download results as CSV, Excel, or chart images |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│         (Chat · Explorer · Export)               │
├─────────────────────────────────────────────────┤
│              Service Layer                       │
│  (Chat · File · Session · Visualization · Export)│
├──────────┬──────────┬──────────┬────────────────┤
│  LLM     │ Sandbox  │Analytics │ Visualization  │
│Groq+     │ Executor │ Engine   │ Engine         │
│Ollama   │ AST Valid│ 8-Stage  │ Chart Selector │
├──────────┴──────────┴──────────┴────────────────┤
│          Core (Config · Security · Logging)      │
├─────────────────────────────────────────────────┤
│         Storage (SQLite · File Manager)          │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### One-Command Docker Launch (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/datawhisperer.git
cd datawhisperer

# Start everything (CPU mode)
docker compose up -d

# Or with GPU acceleration (NVIDIA)
docker compose --profile gpu up -d
```

Open **http://localhost:8501** — that's it.

> First run downloads the AI model (~4 GB). Subsequent starts are instant.

### Using Make

```bash
make docker-up        # CPU mode
make docker-up-gpu    # GPU mode (NVIDIA)
make docker-logs      # Follow startup progress
make health           # Check service status
```

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| [Installation Guide](docs/INSTALL.md) | Detailed setup for all platforms |
| [Deployment Guide](docs/DEPLOY.md) | Production deployment strategies |
| [Troubleshooting Guide](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [Optimization Guide](docs/OPTIMIZATION.md) | Performance tuning for GPU/CPU/memory |

---

## 💻 Native Installation (Without Docker)

### Prerequisites

- **Python 3.11 or 3.12**
- **Ollama** installed and running ([ollama.com](https://ollama.com))

### Steps

```bash
# 1. Install Ollama and pull the model
ollama pull qwen2.5:7b

# 2. Clone and setup
git clone https://github.com/your-org/datawhisperer.git
cd datawhisperer
python -m venv .venv

# Activate venv:
#   Windows:  .venv\Scripts\activate
#   Linux/Mac: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env   # Edit .env if needed

# 5. Run
streamlit run app.py
```

---

## ⚙️ Configuration

All settings are controlled via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | _(empty)_ | Groq API key (primary LLM). Leave empty to use Ollama only |
| `GROK_MODEL` | `llama-3.3-70b-versatile` | Groq model identifier |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint (fallback) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model identifier |
| `OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature |
| `OLLAMA_NUM_CTX` | `4096` | Context window size |
| `SANDBOX_TIMEOUT` | `30` | Code execution timeout (s) |
| `SANDBOX_MAX_MEMORY_MB` | `512` | Sandbox memory limit |
| `STORAGE_MAX_FILE_SIZE_MB` | `50` | Max upload size |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

See [`.env.example`](.env.example) for the complete list.

---

## 🧪 Development

```bash
# Install dev dependencies
make dev

# Run tests with coverage
make test

# Lint + format + typecheck
make qa

# Clean build artifacts
make clean
```

### Project Structure

```
datawhisperer/
├── app.py                  # Streamlit entry point
├── backend/
│   ├── analytics/          # Data profiling, stats, ML, insights
│   ├── core/               # Config, security, logging, exceptions
│   ├── llm/                # LangChain agent, prompts, memory
│   ├── models/             # SQLAlchemy models
│   ├── sandbox/            # Process-isolated code execution
│   ├── services/           # Business logic layer
│   ├── storage/            # File & database management
│   ├── utils/              # CSV analysis, helpers
│   └── visualization/      # Chart selection, generation, themes
├── frontend/
│   ├── components/         # Sidebar, chat, explorer, export
│   ├── state.py            # Streamlit session state management
│   └── theme.py            # Custom CSS injection
├── tests/                  # 22+ test modules, >90% coverage
├── Dockerfile              # Multi-stage production build
├── docker-compose.yml      # Full stack orchestration
├── Makefile                # Uniform command interface
├── pyproject.toml          # PEP 621 project metadata
└── docs/                   # Deployment & operations guides
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

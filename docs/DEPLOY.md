# 🚀 Deployment Guide

Production deployment strategies for DataWhisperer.

---

## Table of Contents

- [Deployment Options](#deployment-options)
- [Docker Compose (Single Server)](#docker-compose-single-server)
- [Reverse Proxy with NGINX](#reverse-proxy-with-nginx)
- [HTTPS with Let's Encrypt](#https-with-lets-encrypt)
- [Cloud Deployment](#cloud-deployment)
- [Security Hardening](#security-hardening)
- [Monitoring](#monitoring)
- [Backup & Recovery](#backup--recovery)

---

## Deployment Options

| Method | Complexity | Best For |
|--------|-----------|----------|
| Docker Compose | Low | Single server, small teams |
| Docker + NGINX | Medium | Production with HTTPS |
| Cloud VM | Medium | Remote access, scalability |
| Kubernetes | High | Enterprise, multi-tenant |

---

## Docker Compose (Single Server)

The simplest production deployment. Suitable for teams of 1–20 users.

### CPU Mode

```bash
# 1. Clone and configure
git clone https://github.com/your-org/datawhisperer.git
cd datawhisperer
cp .env.example .env

# 2. Edit .env for production
# Set APP_DEBUG=false, LOG_LEVEL=WARNING

# 3. Launch
docker compose up -d

# 4. Verify
docker compose ps
curl -f http://localhost:8501/_stcore/health
```

### GPU Mode (NVIDIA)

```bash
# Prerequisites: NVIDIA drivers + nvidia-container-toolkit
# See Optimization Guide for GPU setup details

docker compose --profile gpu up -d
```

### Updating

```bash
# Pull latest code
git pull origin main

# Rebuild and restart (preserves data volumes)
docker compose up -d --build

# Verify
docker compose ps
```

---

## Reverse Proxy with NGINX

For production, place NGINX in front of Streamlit to handle TLS, compression, and WebSocket upgrades.

### docker-compose.override.yml

Create this file alongside your `docker-compose.yml`:

```yaml
services:
  nginx:
    image: nginx:alpine
    container_name: datawhisperer-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
      app:
        condition: service_healthy
    networks:
      - datawhisperer
    restart: unless-stopped
```

### nginx/nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    # Security headers
    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=app:10m rate=10r/s;

    upstream streamlit {
        server app:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect HTTP to HTTPS (enable when certs are ready)
        # return 301 https://$server_name$request_uri;

        client_max_body_size 50M;
        limit_req zone=app burst=20 nodelay;

        location / {
            proxy_pass http://streamlit;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;
        }

        # WebSocket support (required by Streamlit)
        location /_stcore/stream {
            proxy_pass http://streamlit/_stcore/stream;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_read_timeout 86400;
        }
    }
}
```

---

## HTTPS with Let's Encrypt

```bash
# 1. Install certbot
sudo apt-get install certbot

# 2. Generate certificates
sudo certbot certonly --standalone -d your-domain.com

# 3. Copy certs to project
mkdir -p nginx/certs
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/certs/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/certs/

# 4. Update nginx.conf to use SSL
# Add ssl_certificate and ssl_certificate_key directives

# 5. Auto-renew (cron)
echo "0 3 * * * certbot renew --quiet --post-hook 'docker compose restart nginx'" | sudo crontab -
```

---

## Cloud Deployment

### AWS EC2

```bash
# 1. Launch EC2 instance (recommended: t3.xlarge or g4dn.xlarge for GPU)
# 2. SSH into instance
# 3. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 4. Clone and deploy
git clone https://github.com/your-org/datawhisperer.git
cd datawhisperer
cp .env.example .env
docker compose up -d
```

### Google Cloud / Azure

Same Docker Compose workflow applies. Use:
- **GCP**: `e2-standard-4` (CPU) or `n1-standard-4` + T4 GPU
- **Azure**: `Standard_D4s_v3` (CPU) or `Standard_NC4as_T4_v3` (GPU)

---

## Security Hardening

### Production Checklist

- [ ] Set `APP_DEBUG=false` in `.env`
- [ ] Set `LOG_LEVEL=WARNING` or `ERROR`
- [ ] Use NGINX reverse proxy with rate limiting
- [ ] Enable HTTPS with valid certificates
- [ ] Restrict Ollama port (`11434`) to internal network only
- [ ] Set firewall rules — expose only ports 80/443
- [ ] Use Docker secrets for sensitive configuration
- [ ] Keep Docker images updated (`docker compose pull`)
- [ ] Enable log rotation (already configured in the app)

### Restrict Ollama to Internal Access

In `docker-compose.yml`, remove the Ollama port mapping for production:

```yaml
ollama:
  # Remove this line to prevent external access:
  # ports:
  #   - "11434:11434"
```

The app service connects via the Docker network — no host port exposure needed.

---

## Monitoring

### Health Checks

```bash
# Application health
curl -sf http://localhost:8501/_stcore/health && echo "OK" || echo "FAIL"

# Ollama health
curl -sf http://localhost:11434/api/tags && echo "OK" || echo "FAIL"

# Docker service status
docker compose ps --format "table {{.Name}}\t{{.Status}}"
```

### Log Aggregation

Application logs are written to the `app_logs` Docker volume:

```bash
# View live logs
docker compose logs -f app

# Access log files directly
docker exec datawhisperer-app cat /app/logs/app.log
docker exec datawhisperer-app cat /app/logs/error.log
```

---

## Backup & Recovery

### Backup Data Volumes

```bash
#!/bin/bash
# backup.sh — Run periodically via cron
BACKUP_DIR="/backups/datawhisperer/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

for vol in data uploads exports charts; do
  docker run --rm \
    -v "datawhisperer-${vol}:/source:ro" \
    -v "${BACKUP_DIR}:/backup" \
    alpine tar czf "/backup/${vol}.tar.gz" -C /source .
done

echo "Backup complete: $BACKUP_DIR"
```

### Restore

```bash
# Stop services
docker compose down

# Restore a volume
docker run --rm \
  -v "datawhisperer-data:/target" \
  -v "/backups/datawhisperer/20260708_120000:/backup:ro" \
  alpine sh -c "cd /target && tar xzf /backup/data.tar.gz"

# Restart
docker compose up -d
```

---

## Next Steps

- 🔧 [Troubleshooting](TROUBLESHOOTING.md) — Fix common issues
- ⚡ [Optimization](OPTIMIZATION.md) — GPU/CPU tuning

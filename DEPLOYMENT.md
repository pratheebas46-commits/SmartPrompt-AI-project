# SmartPrompt AI — Deployment & Operations Guide

This guide covers running SmartPrompt AI locally in **Offline Mode**, running via **Docker Compose**, and deploying **Online Demo Mode** using free or low-cost cloud hosting.

---

## 1. Localhost Execution (Offline Mode — Preserved)

Offline mode runs with zero internet access using local SQLite and local Scikit-learn models:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Defaults to EXECUTION_MODE="offline"

# Terminal 1 — FastAPI Backend (http://127.0.0.1:8000/docs)
python run_api.py

# Terminal 2 — Streamlit Dashboard (http://127.0.0.1:8501)
python run_app.py
```

---

## 2. Production Hardening & Data Safety

- **No Raw Dataset Persistence in DB:** SQLite/PostgreSQL stores only analysis metadata (`analysis_history` table). Uploaded CSV/Excel files are validated in memory (`_CHUNK_SIZE` capped against `MAX_UPLOAD_SIZE_MB`) and never saved permanently to disk.
- **Automatic Temp & Session Cleanup:**
  - `UploadFile` streams are explicitly closed in a `finally:` block after reading.
  - `DatasetSessionStore` automatically evicts idle sessions after `SESSION_TTL_SECONDS` (default `1800s`) and caps active sessions via `MAX_ACTIVE_SESSIONS` (default `30`).
  - Clients can release memory immediately via `DELETE /api/dataset/{dataset_id}`.
  - FastAPI shutdown hooks clear active sessions and dispose SQLAlchemy pools cleanly.
- **Health & Readiness Endpoints:**
  - `GET /health` — Reports execution mode, database connectivity, online AI flag, and active session count.
  - `GET /ready` — Container readiness probe returning `200 OK` when the database is reachable (`503` otherwise).
  - `GET /docs` — Interactive Swagger API documentation.
- **Cloud-Safe Logging:** Set `ENABLE_FILE_LOGGING=false` in containers so structured logs stream cleanly to `stdout` without requiring writable local log folders.
- **PostgreSQL URL Normalization:** `config/settings.py` automatically converts `postgres://` and `postgresql://` URLs to `postgresql+psycopg2://` for SQLAlchemy 2.0 compatibility.

---

## 3. Docker & Docker Compose Deployment

Run PostgreSQL + FastAPI Backend + Streamlit Frontend together:

```bash
export POSTGRES_PASSWORD="choose-a-strong-password"
docker compose up --build -d
```

- **Backend API & Swagger:** `http://localhost:8000/docs`
- **Backend Health Check:** `http://localhost:8000/health`
- **Frontend Dashboard:** `http://localhost:8501`

To run without Docker Compose (separate containers):

```bash
docker build -f Dockerfile.backend -t smartprompt-backend .
docker build -f Dockerfile.frontend -t smartprompt-frontend .

docker run -d -p 8000:8000 \
  -e EXECUTION_MODE=online \
  -e DATABASE_URL="postgresql+psycopg2://USER:PASS@HOST:5432/smartprompt_ai?sslmode=require" \
  -e CORS_ALLOWED_ORIGINS="http://localhost:8501" \
  smartprompt-backend

docker run -d -p 8501:8501 \
  -e EXECUTION_MODE=online \
  -e API_BASE_URL="http://host.docker.internal:8000" \
  smartprompt-frontend
```

---

## 4. Free / Low-Cost Cloud Hosting Process

### Option A: Render Blueprint (`render.yaml` Included) + PostgreSQL
1. Push this repository to GitHub (verify `.env` is ignored via `.gitignore`).
2. In the Render dashboard, select **New -> Blueprint** and point to your repository (`render.yaml` defines `smartprompt-ai-backend`, `smartprompt-ai-frontend`, and `smartprompt-ai-db`).
3. Wait for `smartprompt-ai-backend` to deploy and copy its public URL (e.g. `https://smartprompt-ai-backend.onrender.com`).
4. Update environment variables in the Render dashboard:
   - On **`smartprompt-ai-frontend`**: set `API_BASE_URL` = backend public URL.
   - On **`smartprompt-ai-backend`**: set `CORS_ALLOWED_ORIGINS` = frontend public URL (e.g. `https://smartprompt-ai-frontend.onrender.com`), and optionally set `AI_PROVIDER=gemini` + `GEMINI_API_KEY` if you want online LLM prompt polishing.
5. Verify `/health` returns `{"status": "healthy", "database_connected": true}` before opening the frontend URL.

### Option B: Split Free Hosting (Streamlit Community Cloud + Render/Railway + Neon PostgreSQL)
1. **Database (Free Tier):** Create a free PostgreSQL database on **Neon** or **Supabase** and copy the connection string (`postgresql://...?sslmode=require`).
2. **Backend (Free/Low-Cost):** Deploy `Dockerfile.backend` (or `python run_api.py`) to **Render** or **Railway** with `EXECUTION_MODE=online`, `DATABASE_URL=<your-db-url>`, `ENABLE_FILE_LOGGING=false`.
3. **Frontend (Free):** Deploy `app.py` on **Streamlit Community Cloud** and set `API_BASE_URL` in Streamlit Secrets to your deployed backend URL. Set `CORS_ALLOWED_ORIGINS` on the backend to your Streamlit app URL.

### Known Free-Tier Platform Limits
- **Cold Starts:** Free web services (e.g., Render Free) sleep after ~15 minutes of inactivity, causing a 30–60 second wake-up delay on the first request.
- **Ephemeral Filesystems:** Free container disks reset on redeploy/restart; always use an external PostgreSQL `DATABASE_URL` in online mode to preserve analysis history.
- **Memory & Upload Size:** Free containers typically provide 512 MB RAM; keep `MAX_UPLOAD_SIZE_MB` at `25` or lower in production.

---

## 5. Post-Deployment Verification Checklist

Run these checks against your live deployed URLs:
1. `curl -s https://<your-backend-url>/health` -> Expect HTTP 200 with `"status": "healthy"` and `"database_connected": true`.
2. Open `https://<your-backend-url>/docs` -> Confirm Swagger UI loads.
3. Open `https://<your-frontend-url>` -> Confirm sidebar shows **Backend Connected (ONLINE | postgresql+psycopg2)**.
4. Upload a sample CSV, optimize a prompt, run analysis + validation + anomaly detection, verify history entry persists across page refresh, and download both CSV and PDF reports.

> **Note:** No cloud deployment has been executed from this local workspace container; follow Section 3 or Section 4 with your hosting credentials to obtain and verify your live public demo URL.

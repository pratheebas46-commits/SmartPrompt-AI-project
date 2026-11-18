# SmartPrompt AI project
AI powered data analytics and prompt optimization platform
# SmartPrompt AI: A Dataset-Aware, Self-Validating Intelligent Data Analyst

SmartPrompt AI is an intermediate-level Python AI/ML and data analytics platform that accepts CSV and Excel datasets, detects analytical intent from natural-language questions, optimizes prompts using dataset schema context, scores prompt quality using Scikit-learn, performs deterministic exploratory data analysis, detects multivariate anomalies with Isolation Forest, cross-validates results against raw data calculations, and delivers interactive Plotly visualizations and CSV/PDF reports through a FastAPI + Streamlit architecture.

---

## Dual Execution Modes

### 1. Offline Local Mode (`EXECUTION_MODE=offline`)
- Runs 100% on `localhost` with **zero internet or external API dependency**.
- Local Streamlit UI (`127.0.0.1:8501`) + Local FastAPI backend (`127.0.0.1:8000`).
- Local SQLite persistence (`data/smartprompt_ai.db`) storing analysis metadata only.
- Local Scikit-learn / Joblib models (`TF-IDF + LogisticRegression` & `IsolationForest`).
- Deterministic Pandas / NumPy execution and self-validation.

### 2. Online Demo Mode (`EXECUTION_MODE=online`)
- Cloud-ready `0.0.0.0` bindings and ephemeral temp directories (does not require persistent local repo paths).
- Configurable PostgreSQL database via `DATABASE_URL` (auto-normalizes `postgres://` and `postgresql://`) with SQLite fallback unless `STRICT_ONLINE_MODE=true`.
- Optional server-side online LLM rewrite (`GEMINI_API_KEY`) strictly verified against dataset schema columns, with automatic deterministic fallback.
- Environment-configured `CORS_ALLOWED_ORIGINS` and `API_BASE_URL`.

---

## Project Architecture

```text
.
├── app.py                        # Module 13: Streamlit Interactive Dashboard
├── run_api.py                    # FastAPI Uvicorn Launcher (localhost & cloud $PORT)
├── run_app.py                    # Streamlit Launcher (localhost & cloud $PORT)
├── config/
│   ├── settings.py               # Module 14: Environment & Offline/Online Settings
│   ├── mode_manager.py           # Module 14: Mode Capabilities & Fallback Resolver
│   └── logging_config.py         # Console + Optional Rotating File Logging
├── modules/
│   ├── dataset_profiler.py       # Module 1: CSV/Excel Ingestion & Profiling
│   ├── data_cleaner.py           # Module 2: Non-Destructive Cleaning & Preprocessing
│   ├── intent_detector.py        # Module 3: NLP + Rule-Based Intent Detection
│   ├── prompt_optimizer.py       # Module 4: Dataset-Aware Prompt Optimizer
│   ├── prompt_quality_model.py   # Module 5: ML Prompt Quality Evaluator (Scikit-learn)
│   ├── analysis_engine.py        # Module 6: Deterministic Data Analysis Engine
│   ├── anomaly_detector.py       # Module 7: ML Anomaly Detection (Isolation Forest)
│   ├── validator.py              # Module 8: Evidence-Based Self-Validation Engine
│   ├── visualization.py          # Module 9: Interactive Plotly Visualization Engine
│   └── report_generator.py       # Module 11: CSV & ReportLab PDF Exporter
├── database/
│   └── db_manager.py             # Module 10: SQLAlchemy History Manager (SQLite/PostgreSQL)
├── backend/
│   ├── main.py                   # Module 12: FastAPI App, CORS, Error Handlers & Docs
│   ├── routes/                   # REST Endpoints (/health, /api/dataset, /api/prompt, /api/analysis, /api/history)
│   ├── schemas/                  # Pydantic v2 Request & Response Models
│   └── services/                 # AnalyticsService & TTL DatasetSessionStore
├── frontend/
│   └── api_client.py             # Decoupled HTTP Client for FastAPI
├── tests/                        # Unit & Integration Pytest Suites
├── Dockerfile.backend            # Backend Production Container
├── Dockerfile.frontend           # Frontend Production Container
├── docker-compose.yml            # Multi-Container Stack (Postgres + FastAPI + Streamlit)
├── render.yaml                   # Render Free/Low-Cost Deployment Blueprint
├── .env.example                  # Local Offline Configuration Template
├── .env.production.example       # Online Deployment Configuration Template
├── DEPLOYMENT.md                 # Full Deployment & Hosting Documentation
└── requirements.txt              # Python Dependencies
```

---

## Configuration Variables

| Variable | Offline Default | Online Default | Description |
| :--- | :--- | :--- | :--- |
| `EXECUTION_MODE` | `offline` | `online` | Selects `offline` or `online` execution profile. |
| `STRICT_ONLINE_MODE` | `false` | `false` | If `true`, fails fast if `DATABASE_URL` or `GEMINI_API_KEY` is missing in online mode. |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | `0.0.0.0` / `$PORT` | FastAPI bind host and port. |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Backend public URL | Target FastAPI URL used by the Streamlit frontend. |
| `STREAMLIT_HOST` / `STREAMLIT_PORT` | `127.0.0.1` / `8501` | `0.0.0.0` / `$PORT` | Streamlit server bind host and port. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:8501,...` | Frontend public URL(s) | Comma-separated origins allowed by FastAPI CORS middleware. |
| `SQLITE_DB_PATH` | `data/smartprompt_ai.db` | `<temp>/smartprompt_ai.db` | SQLite path used offline or as fallback. |
| `DATABASE_URL` | `""` | `postgresql+psycopg2://...` | Remote PostgreSQL connection string for online mode. |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` | `5` / `10` | `5` / `10` | SQLAlchemy pool parameters for PostgreSQL. |
| `AI_PROVIDER` / `GEMINI_API_KEY` | `local` / `""` | `local` or `gemini` | Optional server-side online LLM rewrite credentials (never exposed to frontend). |
| `MAX_UPLOAD_SIZE_MB` | `50` | `25`–`50` | Maximum allowed CSV/Excel upload size. |
| `SESSION_TTL_SECONDS` / `MAX_ACTIVE_SESSIONS` | `1800` / `30` | `1800` / `30` | Automatic in-memory upload session expiry and cap. |
| `ENABLE_FILE_LOGGING` / `LOG_LEVEL` | `true` / `INFO` | `false` / `INFO` | Rotating file log toggle (`stdout` is always active). |

---

## Localhost Quickstart (Offline Mode)

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Create `.env`:**
   ```bash
   cp .env.example .env
   ```
4. **Run the FastAPI backend (Terminal 1):**
   ```bash
   python run_api.py
   # Swagger UI: http://127.0.0.1:8000/docs
   ```
5. **Run the Streamlit frontend (Terminal 2):**
   ```bash
   python run_app.py
   # Dashboard: http://127.0.0.1:8501
   ```
6. **Run the complete test suite:**
   ```bash
   pytest tests/ -v
   ```

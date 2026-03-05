# research-poc backend

This folder contains a minimal FastAPI backend for the research POC frontend. It
provides:

- REST endpoints for patient CRUD and multi-step forms (demographics, lifestyle,
  symptoms/exams)
- Wearable file upload (CSV/JSON) with metadata storage and lightweight parsing
- SQLAlchemy models and DB initialization (defaults to SQLite for local dev)

Quick start

1. Create and activate a Python virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate    # on Windows use: .venv\\Scripts\\activate
pip install -r s.txt
```

2. Run the app (defaults to SQLite `backend.db` in this folder)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Endpoints

- POST /api/v1/patients — create a patient
- GET /api/v1/patients — list patients
- GET /api/v1/patients/{id} — retrieve single patient with related sections
- PUT /api/v1/patients/{id}/demographics — save demographics
- PUT /api/v1/patients/{id}/lifestyle — save lifestyle
- POST /api/v1/patients/{id}/notes — add a clinical note (symptoms/mental/exams)
- POST /api/v1/patients/{id}/wearable/upload — upload file (multipart form,
  field `file`)

4. Test examples (after server is running)

Create patient:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/patients" -H "Content-Type: application/json" \
  -d '{"name":"John Doe","age":45,"gender":"male","weight":82.5,"height":180}'
```

Upload wearable (multipart):

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/patients/<ID>/wearable/upload" \
  -F "file=@/path/to/wearable_data.json"
```

Notes

- The app creates tables automatically on startup using SQLAlchemy
  `Base.metadata.create_all(engine)` for convenience in dev. For production,
  replace with Alembic migrations (alembic is included in `s.txt`).
- Default DB is SQLite at `sqlite:///./backend.db`. To use Postgres, set
  environment variable `DATABASE_URL` to e.g.
  `postgresql+psycopg2://user:pass@host:5432/dbname` and ensure appropriate DB
  driver is installed (`psycopg2-binary`).

If you want, I can:

- generate Alembic migration scripts
- add authentication and role-based access
- add background parsing (Celery)

Choose the next step and I'll implement it.

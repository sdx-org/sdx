**Summary** Replaces ad-hoc `print()` with `logging` + `RichHandler` in
`research/backend/cli.py`, exposes a safe preview `--dry-run` in
`scripts/gen_models/gen_sqla.py`, and adds a brief GSOC onboarding note to
`DEVELOPMENT.md`.

**Files changed**

- `research/backend/cli.py`
- `scripts/gen_models/gen_sqla.py`
- `DEVELOPMENT.md`

**Why**

- Low-risk, high-ROI changes to improve developer/contributor experience and
  create a clear GSOC-friendly starter task.

**How to test (smoke)**

- `python -c "import importlib; m=importlib.import_module('research.backend.cli'); print('IMPORT_OK', bool(getattr(m,'logger',None)))"`
- `python -c "from research.backend import cli; cli.logger.info('demo')"`
- `python -m research.backend.cli --help`
- `python scripts/gen_models/gen_sqla.py --dry-run --output preview.py`

**Acceptance criteria**

- Smoke checks above succeed.
- Small focused commits with clear messages.

**Notes for reviewers**

- Confirm logging output is clear and backward-compatible for automated scripts.
- Confirm `--dry-run` prints preview and does not create files.

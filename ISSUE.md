**Title**: Improve CLI UX: structured logging, `--dry-run` generator, and GSOC onboarding docs

**Motivation**
- Small, low-risk changes that improve contributor experience and make a perfect GSOC starter task.

**What to change**
- Replace ad-hoc `print()` calls with structured `logging` + `RichHandler` in `research/backend/cli.py`.
- Add `--dry-run` (preview mode) and `--output` to `scripts/gen_models/gen_sqla.py` so generated code can be previewed without writing files.
- Add a short GSOC / mentorship section to `DEVELOPMENT.md` describing how to take this issue as a starter task.

**How to test (smoke)**
- `python -c "import importlib; m=importlib.import_module('research.backend.cli'); print('IMPORT_OK', bool(getattr(m,'logger',None)))"`
- `python -c "from research.backend import cli; cli.logger.info('demo')"`
- `python -m research.backend.cli --help`
- `python scripts/gen_models/gen_sqla.py --dry-run --output preview.py`

**Acceptance criteria**
- Changes applied in 2–3 small commits.
- Smoke checks above run and show expected output.
- `DEVELOPMENT.md` includes a short note linking the issue.
- PR links to this issue and is labeled `good first issue` / `gsoc`.

**Labels**: `good first issue`, `enhancement`, `gsoc`

**Notes for mentors**
- Review logging formatting and ensure no behavior regressions in non-interactive automation.
- Verify `--dry-run` outputs a clear preview and does not write files when requested.

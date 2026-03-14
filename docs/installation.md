# Installation

## Stable release

Install from PyPI:

```bash
pip install hiperhealth
```

`hiperhealth` requires Python `>=3.10,<4`.

## System dependencies

Some extraction features rely on system packages:

- `tesseract` for OCR on image-based reports
- `libmagic` for MIME type detection

The conda development environment already includes them.

## From source

Clone the repository:

```bash
git clone https://github.com/hiperhealth/hiperhealth
cd hiperhealth
```

Create the development environment:

```bash
conda env create -f conda/dev.yaml -n hiperhealth
conda activate hiperhealth
```

Install the package and development tooling:

```bash
./scripts/install-dev.sh
```

## Verify the installation

Run the test suite:

```bash
pytest -vv
```

Build the docs locally:

```bash
mkdocs serve --watch docs --config-file mkdocs.yaml
```

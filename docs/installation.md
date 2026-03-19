# Installation

## From PyPI

```bash
pip install hiperhealth
```

!!! info "Python version" `hiperhealth` requires **Python >= 3.10, < 4**.

## System Dependencies

Some extraction features rely on system packages:

| Package     | Purpose                            |
| ----------- | ---------------------------------- |
| `tesseract` | OCR on image-based medical reports |
| `libmagic`  | MIME type detection                |

!!! tip The conda development environment already includes these.

## From Source

Clone the repository and create the development environment:

```bash
git clone https://github.com/hiperhealth/hiperhealth
cd hiperhealth
```

```bash
conda env create -f conda/dev.yaml -n hiperhealth
conda activate hiperhealth
```

Install the package and development tooling:

```bash
./scripts/install-dev.sh
```

## Verify the Installation

Run the test suite:

```bash
pytest -vv
```

Build and preview the docs locally:

```bash
mkdocs serve --watch docs --config-file mkdocs.yaml
```

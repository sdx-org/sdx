# Installation

## From PyPI

```bash
pip install hiperhealth
```

!!! info "Python version" `hiperhealth` requires **Python >= 3.10, < 4**.

## Notebook UI Extra

The interactive notebook interface is optional so base installations do not pull
in Jupyter widget dependencies:

```bash
pip install "hiperhealth[notebook]"
```

Use it from a notebook with:

```python
from hiperhealth.notebook import ui

ui.show()
ui.show(data_dir="/my/data/path")
```

The import is safe in a base install, but calling `ui.show()` requires the
optional extra.

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

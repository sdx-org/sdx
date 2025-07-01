"""Formatting tools."""

import shutil
import subprocess

from pathlib import Path


def run_ruff(path: Path, *, fix: bool = True) -> None:
    """
    Format / lint the generated file with Ruff.

    Parameters
    ----------
        path: Absolute path to the file that was just written.
        fix:  If True, run Ruff with `--fix`; otherwise `check` only.

    Raises
    ------
        RuntimeError: If Ruff is not installed or returns a non-zero exit code.
    """
    ruff_exe = shutil.which('ruff')
    if ruff_exe is None:
        raise RuntimeError(
            'Ruff executable not found. '
            'Install with `pip install ruff` or add it to your PATH.'
        )

    cmd = [ruff_exe, 'format', str(path)]

    try:
        subprocess.run(cmd, check=True)
        print(f'[✓] Ruff {"fixed" if fix else "checked"} {path}')
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'Ruff reported issues (exit code {exc.returncode}).'
        ) from exc

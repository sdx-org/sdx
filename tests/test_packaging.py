"""
title: Packaging regression tests.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile

from pathlib import Path
from typing import Any


def test_wheel_contains_skill_manifests(tmp_path: Any) -> None:
    """
    title: Verify that the built wheel includes skill.yaml manifests.
    parameters:
      tmp_path:
        type: Any
    """
    project_root = Path(__file__).resolve().parent.parent

    # Build the wheel into tmp_path
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'build',
            '--wheel',
            '--outdir',
            str(tmp_path),
            str(project_root),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f'Wheel build failed:\n{result.stderr}\n{result.stdout}'
    )

    wheels = list(tmp_path.glob('*.whl'))
    assert len(wheels) == 1, 'Expected exactly one wheel to be built'
    wheel_path = wheels[0]

    with zipfile.ZipFile(wheel_path) as zf:
        namelist = zf.namelist()

    # The built-in skills that should have a skill.yaml
    expected_skills = ['diagnostics', 'extraction', 'privacy']

    for skill in expected_skills:
        expected_manifest = f'hiperhealth/skills/{skill}/skill.yaml'
        assert expected_manifest in namelist, (
            f"Expected manifest '{expected_manifest}' not found in wheel. "
            f'Found: {namelist}'
        )

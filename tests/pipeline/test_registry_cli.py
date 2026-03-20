"""
title: CLI tests for the channel-aware registry.
"""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from hiperhealth.cli import main

from ._registry_test_utils import (
    bump_channel_skill_version,
    create_channel_repo,
)


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """
    Create a temporary CLI registry directory.

    Parameters
    ----------
    tmp_path:
        Temporary directory provided by pytest.

    Returns
    -------
    Path
        Registry artifact directory.
    """
    return tmp_path / '.hiperhealth' / 'artifacts' / 'skills'


@pytest.fixture
def channel_repo(tmp_path: Path) -> Path:
    """
    Create a local git-backed channel fixture.

    Parameters
    ----------
    tmp_path:
        Temporary directory provided by pytest.

    Returns
    -------
    Path
        Channel repository path.
    """
    return create_channel_repo(tmp_path)


def run_cli(
    registry_dir: Path,
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> str:
    """
    Invoke the CLI and return stdout.

    Parameters
    ----------
    registry_dir:
        Registry artifact directory.
    capsys:
        Pytest capture fixture.
    args:
        CLI arguments after ``--registry-dir``.

    Returns
    -------
    str
        Captured stdout.
    """
    exit_code = main(['--registry-dir', str(registry_dir), *args])
    captured = capsys.readouterr()
    assert exit_code == 0
    return captured.out.strip()


def test_channel_add(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Add a channel through the CLI.
    """
    stdout = run_cli(
        registry_dir,
        capsys,
        'channel',
        'add',
        str(channel_repo),
        '--name',
        'tm',
    )

    assert stdout == 'tm'


def test_channel_list(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    List registered channels through the CLI.
    """
    run_cli(
        registry_dir,
        capsys,
        'channel',
        'add',
        str(channel_repo),
        '--name',
        'tm',
    )

    stdout = run_cli(registry_dir, capsys, 'channel', 'list')
    payload = json.loads(stdout)

    assert payload[0]['local_name'] == 'tm'


def test_channel_skills(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    List channel skills through the CLI.
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))

    stdout = run_cli(registry_dir, capsys, 'channel', 'skills', 'tm')
    payload = json.loads(stdout)

    assert [item['canonical_id'] for item in payload] == [
        'tm.ayurveda',
        'tm.nutrition',
        'tm.triage',
    ]


def test_skill_list(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    List available skills through the CLI.
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))

    stdout = run_cli(
        registry_dir,
        capsys,
        'skill',
        'list',
        '--channel',
        'tm',
    )
    payload = json.loads(stdout)

    assert [item['canonical_id'] for item in payload] == [
        'tm.ayurveda',
        'tm.nutrition',
        'tm.triage',
    ]


def test_channel_install(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Install all channel skills through the CLI.
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))

    stdout = run_cli(
        registry_dir,
        capsys,
        'channel',
        'install',
        'tm',
        '--all',
    )
    payload = json.loads(stdout)

    assert payload == ['tm.ayurveda', 'tm.nutrition']


def test_skill_install(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Install one skill through the CLI.
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))

    stdout = run_cli(
        registry_dir,
        capsys,
        'skill',
        'install',
        'tm.ayurveda',
    )

    assert stdout == 'tm.ayurveda'


def test_skill_update(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Update one installed skill through the CLI.
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))
    run_cli(registry_dir, capsys, 'skill', 'install', 'tm.ayurveda')
    bump_channel_skill_version(channel_repo, 'ayurveda', '0.2.0')

    stdout = run_cli(
        registry_dir,
        capsys,
        'skill',
        'update',
        'tm.ayurveda',
        '--pull',
    )
    listing = json.loads(
        run_cli(
            registry_dir,
            capsys,
            'skill',
            'list',
            '--channel',
            'tm',
            '--installed-only',
        )
    )

    assert stdout == 'tm.ayurveda'
    assert listing[0]['version'] == '0.2.0'


def test_skill_remove(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Remove one installed skill through the CLI.
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))
    run_cli(registry_dir, capsys, 'skill', 'install', 'tm.ayurveda')

    stdout = run_cli(
        registry_dir,
        capsys,
        'skill',
        'remove',
        'tm.ayurveda',
    )
    listing = json.loads(
        run_cli(
            registry_dir,
            capsys,
            'skill',
            'list',
            '--channel',
            'tm',
            '--installed-only',
        )
    )

    assert stdout == 'tm.ayurveda'
    assert listing == []


def test_channel_remove(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Remove a channel through the CLI.
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))
    run_cli(registry_dir, capsys, 'skill', 'install', 'tm.ayurveda')

    stdout = run_cli(registry_dir, capsys, 'channel', 'remove', 'tm')
    payload = json.loads(run_cli(registry_dir, capsys, 'channel', 'list'))

    assert stdout == 'tm'
    assert payload == []

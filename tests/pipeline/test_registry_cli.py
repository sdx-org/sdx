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
    title: Provide a temporary registry directory for CLI tests.
    parameters:
      tmp_path:
        type: Path
    returns:
      type: Path
    """
    return tmp_path / '.hiperhealth' / 'artifacts' / 'skills'


@pytest.fixture
def channel_repo(tmp_path: Path) -> Path:
    """
    title: Create a channel repository fixture for CLI tests.
    parameters:
      tmp_path:
        type: Path
    returns:
      type: Path
    """
    return create_channel_repo(tmp_path)


def run_cli(
    registry_dir: Path,
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> str:
    """
    title: Run the CLI main entrypoint and capture stdout.
    parameters:
      registry_dir:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
      args:
        type: str
        variadic: positional
    returns:
      type: str
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
    title: The CLI should register a channel and print its alias.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
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
    title: The CLI should list registered channels as JSON.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
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
    title: The CLI should list skills declared by one channel.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))

    stdout = run_cli(registry_dir, capsys, 'channel', 'skills', 'tm')
    payload = json.loads(stdout)

    assert [item['canonical_id'] for item in payload] == [
        'tm.ayurveda',
        'tm.nutrition',
        'tm.triage',
    ]


def test_channel_update(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    title: The CLI should refresh a channel and installed skills.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))
    run_cli(registry_dir, capsys, 'skill', 'install', 'tm.ayurveda')
    bump_channel_skill_version(channel_repo, 'ayurveda', '0.2.0')

    stdout = run_cli(registry_dir, capsys, 'channel', 'update', 'tm')
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

    assert json.loads(stdout) == ['tm.ayurveda']
    assert listing[0]['version'] == '0.2.0'


def test_skill_list(
    registry_dir: Path,
    channel_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    title: The CLI should list skills filtered by channel.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
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
    title: The CLI should install all enabled skills from a channel.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
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
    title: The CLI should install one canonical channel skill id.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
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
    title: The CLI should update one installed channel skill.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
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
    title: The CLI should remove one installed channel skill.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
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
    title: The CLI should remove a registered channel.
    parameters:
      registry_dir:
        type: Path
      channel_repo:
        type: Path
      capsys:
        type: pytest.CaptureFixture[str]
    """
    run_cli(registry_dir, capsys, 'channel', 'add', str(channel_repo))
    run_cli(registry_dir, capsys, 'skill', 'install', 'tm.ayurveda')

    stdout = run_cli(registry_dir, capsys, 'channel', 'remove', 'tm')
    payload = json.loads(run_cli(registry_dir, capsys, 'channel', 'list'))

    assert stdout == 'tm'
    assert payload == []

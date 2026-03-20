"""
title: Channel-aware registry tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hiperhealth.pipeline import (
    PipelineContext,
    SkillRegistry,
    Stage,
    StageRunner,
)
from pydantic import ValidationError

from ._registry_test_utils import (
    bump_channel_skill_version,
    create_channel_repo,
    create_legacy_repo,
    write_file,
)


@pytest.fixture
def registry(tmp_path: Path) -> SkillRegistry:
    """
    Create a registry rooted in a temporary directory.

    Parameters
    ----------
    tmp_path:
        Temporary directory provided by pytest.

    Returns
    -------
    SkillRegistry
        Test registry.
    """
    return SkillRegistry(
        registry_dir=tmp_path / '.hiperhealth' / 'artifacts' / 'skills'
    )


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


@pytest.fixture
def legacy_repo(tmp_path: Path) -> Path:
    """
    Create a legacy single-skill repository fixture.

    Parameters
    ----------
    tmp_path:
        Temporary directory provided by pytest.

    Returns
    -------
    Path
        Legacy repository path.
    """
    return create_legacy_repo(tmp_path)


def test_parse_valid_skills_yaml(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Parse a valid channel manifest.
    """
    manifest = registry._read_channel_manifest(channel_repo)

    assert manifest.channel.name == 'traditional-medicine'
    assert manifest.channel.default_alias == 'tm'
    assert [skill.name for skill in manifest.skills] == [
        'ayurveda',
        'nutrition',
        'triage',
    ]


def test_reject_invalid_skills_yaml(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Reject an invalid channel manifest.
    """
    write_file(
        channel_repo / 'skills.yaml',
        """
        api_version: 1
        skills: []
        """,
    )

    with pytest.raises(ValidationError):
        registry._read_channel_manifest(channel_repo)


def test_reject_duplicate_skill_names_within_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Reject duplicate skill names within one channel manifest.
    """
    write_file(
        channel_repo / 'skills.yaml',
        """
        api_version: 1
        channel:
          name: traditional-medicine
          default_alias: tm
        skills:
          - name: ayurveda
            path: skills/ayurveda
            manifest: skills/ayurveda/hiperhealth.yaml
          - name: ayurveda
            path: skills/nutrition
            manifest: skills/nutrition/hiperhealth.yaml
        """,
    )

    with pytest.raises(ValueError, match='Duplicate skill names'):
        registry._read_channel_manifest(channel_repo)


def test_reject_missing_per_skill_manifest(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Reject channel manifests that reference missing skill manifests.
    """
    (channel_repo / 'skills' / 'nutrition' / 'hiperhealth.yaml').unlink()

    with pytest.raises(ValueError, match='Declared manifest'):
        registry._read_channel_manifest(channel_repo)


def test_enforce_local_alias_uniqueness(
    registry: SkillRegistry,
    tmp_path: Path,
) -> None:
    """
    Reject duplicate local aliases.
    """
    first_repo = create_channel_repo(tmp_path / 'first')
    second_repo = create_channel_repo(tmp_path / 'second')

    assert registry.add_channel(str(first_repo), local_name='tm') == 'tm'
    with pytest.raises(ValueError, match='already registered'):
        registry.add_channel(str(second_repo), local_name='tm')


def test_list_channels_and_skills(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    List registered channels and available skills.
    """
    registry.add_channel(str(channel_repo))

    channels = registry.list_channels()
    assert [channel.local_name for channel in channels] == ['tm']

    channel_skills = registry.list_channel_skills('tm')
    assert [skill.canonical_id for skill in channel_skills] == [
        'tm.ayurveda',
        'tm.nutrition',
        'tm.triage',
    ]

    all_skills = registry.list_skills()
    canonical_ids = {skill.canonical_id for skill in all_skills}
    assert 'hiperhealth.privacy' in canonical_ids
    assert 'tm.ayurveda' in canonical_ids
    assert 'tm.nutrition' in canonical_ids
    assert 'tm.triage' in canonical_ids

    tm_skills = registry.list_skills(channel='tm')
    assert [skill.canonical_id for skill in tm_skills] == [
        'tm.ayurveda',
        'tm.nutrition',
        'tm.triage',
    ]


def test_register_channel_from_local_git_fixture(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Register a channel from a local git repository.
    """
    local_name = registry.add_channel(str(channel_repo))

    assert local_name == 'tm'
    assert (
        registry.root_dir / 'channels' / 'tm' / 'repo' / 'skills.yaml'
    ).exists()
    channel = registry.list_channels()[0]
    assert channel.source == str(channel_repo)
    assert channel.remote_name == 'traditional-medicine'
    assert channel.provider == 'local'


def test_install_one_skill_from_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Install one skill from a channel.
    """
    registry.add_channel(str(channel_repo))
    installed = registry.install_skill('tm.ayurveda')

    assert installed == 'tm.ayurveda'
    state = registry._load_state()
    assert state.skills['tm.ayurveda'].manifest_path.endswith(
        'skills/ayurveda/hiperhealth.yaml'
    )

    skill = registry.load('tm.ayurveda')
    assert skill.metadata.name == 'tm.ayurveda'
    assert skill.metadata.version == '0.1.0'


def test_install_all_skills_from_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Install all enabled skills from a channel.
    """
    registry.add_channel(str(channel_repo))

    installed = registry.install_channel('tm')

    assert installed == ['tm.ayurveda', 'tm.nutrition']
    installed_only = registry.list_skills(channel='tm', installed_only=True)
    assert [skill.canonical_id for skill in installed_only] == installed


def test_install_channel_include_disabled(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Install all channel skills including disabled ones.
    """
    registry.add_channel(str(channel_repo))

    installed = registry.install_channel('tm', include_disabled=True)

    assert installed == ['tm.ayurveda', 'tm.nutrition', 'tm.triage']


def test_update_one_skill_without_pulling_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Refresh one skill without pulling the channel checkout.
    """
    registry.add_channel(str(channel_repo))
    registry.install_skill('tm.ayurveda')
    bump_channel_skill_version(channel_repo, 'ayurveda', '0.2.0')

    registry.update_skill('tm.ayurveda', pull_channel=False)

    state = registry._load_state()
    assert state.skills['tm.ayurveda'].version == '0.1.0'


def test_update_one_skill_with_pulling_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Refresh one skill after pulling the channel checkout.
    """
    registry.add_channel(str(channel_repo))
    registry.install_skill('tm.ayurveda')
    bump_channel_skill_version(channel_repo, 'ayurveda', '0.2.0')

    registry.update_skill('tm.ayurveda', pull_channel=True)

    state = registry._load_state()
    assert state.skills['tm.ayurveda'].version == '0.2.0'
    skill = registry.load('tm.ayurveda')
    assert skill.metadata.version == '0.2.0'


def test_update_whole_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Update a channel checkout and refresh installed skills.
    """
    registry.add_channel(str(channel_repo))
    registry.install_skill('tm.ayurveda')
    before_commit = registry.list_channels()[0].commit
    bump_channel_skill_version(channel_repo, 'ayurveda', '0.3.0')

    updated = registry.update_channel('tm')

    channel = registry.list_channels()[0]
    assert updated == ['tm.ayurveda']
    assert channel.commit != before_commit
    assert registry._load_state().skills['tm.ayurveda'].version == '0.3.0'


def test_remove_one_skill(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Remove one installed channel skill.
    """
    registry.add_channel(str(channel_repo))
    registry.install_skill('tm.ayurveda')

    registry.remove_skill('tm.ayurveda')

    with pytest.raises(KeyError, match='available but not installed'):
        registry.load('tm.ayurveda')


def test_remove_whole_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Remove a channel and its installed skills.
    """
    registry.add_channel(str(channel_repo))
    registry.install_skill('tm.ayurveda')

    registry.remove_channel('tm')

    assert registry.list_channels() == []
    assert not (registry.root_dir / 'channels' / 'tm').exists()
    assert 'tm.ayurveda' not in registry._load_state().skills


def test_load_skill_by_canonical_id_and_stage_runner(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    Load a channel skill by canonical id and run it through the runner.
    """
    registry.add_channel(str(channel_repo))
    registry.install_skill('tm.nutrition')

    runner = StageRunner(registry=registry)
    runner.register('tm.nutrition')
    ctx = runner.run(Stage.TREATMENT, PipelineContext(patient={}))

    assert runner.skills[0].metadata.name == 'tm.nutrition'
    assert ctx.extras['nutrition'] == 'fiber first'


def test_legacy_single_skill_repo_install_still_works(
    registry: SkillRegistry,
    legacy_repo: Path,
) -> None:
    """
    Preserve legacy single-skill repository installs.
    """
    installed = registry.install(str(legacy_repo))

    assert installed == 'legacy.greeting'
    skill = registry.load('legacy.greeting')
    assert skill.metadata.name == 'legacy.greeting'
    assert any(
        summary.canonical_id == 'legacy.greeting'
        for summary in registry.list_skills(installed_only=True)
    )

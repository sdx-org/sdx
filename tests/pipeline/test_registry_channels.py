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
    write_file,
)


@pytest.fixture
def registry(tmp_path: Path) -> SkillRegistry:
    """
    title: Provide a temporary channel-aware registry instance.
    parameters:
      tmp_path:
        type: Path
    returns:
      type: SkillRegistry
    """
    return SkillRegistry(
        registry_dir=tmp_path / '.hiperhealth' / 'artifacts' / 'skills'
    )


@pytest.fixture
def channel_repo(tmp_path: Path) -> Path:
    """
    title: Create a Git-backed channel fixture repository.
    parameters:
      tmp_path:
        type: Path
    returns:
      type: Path
    """
    return create_channel_repo(tmp_path)


@pytest.fixture
def channel_folder(tmp_path: Path) -> Path:
    """
    title: Create a plain local folder channel fixture.
    parameters:
      tmp_path:
        type: Path
    returns:
      type: Path
    """
    return create_channel_repo(tmp_path / 'folder-source', use_git=False)


def test_parse_valid_skills_channel_yaml(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    title: Valid channel manifests should parse successfully.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
    """
    manifest = registry._read_channel_manifest(channel_repo)

    assert manifest.channel.name == 'traditional-medicine'
    assert manifest.channel.default_alias == 'tm'
    assert [skill.name for skill in manifest.skills] == [
        'ayurveda',
        'nutrition',
        'triage',
    ]


def test_reject_invalid_skills_channel_yaml(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    title: Invalid channel manifests should fail validation.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
    """
    write_file(
        channel_repo / 'skills-channel.yaml',
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
    title: Duplicate declared skill names should be rejected.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
    """
    write_file(
        channel_repo / 'skills-channel.yaml',
        """
        api_version: 1
        channel:
          name: traditional-medicine
          default_alias: tm
        skills:
          - name: ayurveda
          - name: ayurveda
        """,
    )

    with pytest.raises(ValueError, match='Duplicate skill names'):
        registry._read_channel_manifest(channel_repo)


def test_reject_missing_per_skill_manifest(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    title: Missing per-skill manifests should fail channel validation.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
    """
    (channel_repo / 'skills' / 'nutrition' / 'skill.yaml').unlink()

    with pytest.raises(ValueError, match='expected manifest'):
        registry._read_channel_manifest(channel_repo)


def test_enforce_local_alias_uniqueness(
    registry: SkillRegistry,
    tmp_path: Path,
) -> None:
    """
    title: Channel aliases must remain unique in local state.
    parameters:
      registry:
        type: SkillRegistry
      tmp_path:
        type: Path
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
    title: Listing APIs should expose channels and their skills.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
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
    title: Git-backed local folders should register as local channels.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
    """
    local_name = registry.add_channel(str(channel_repo))

    assert local_name == 'tm'
    assert (
        registry.root_dir / 'channels' / 'tm' / 'repo' / 'skills-channel.yaml'
    ).exists()
    channel = registry.list_channels()[0]
    assert channel.source == str(channel_repo)
    assert channel.remote_name == 'traditional-medicine'
    assert channel.provider == 'local'


def test_register_channel_from_plain_local_folder(
    registry: SkillRegistry,
    channel_folder: Path,
) -> None:
    """
    title: Plain local folders should also register as channels.
    parameters:
      registry:
        type: SkillRegistry
      channel_folder:
        type: Path
    """
    local_name = registry.add_channel(str(channel_folder))

    channel = registry.list_channels()[0]
    assert local_name == 'tm'
    assert channel.provider == 'local'
    assert channel.source == str(channel_folder.resolve())
    assert channel.commit == ''


def test_reject_ref_for_local_folder_channel(
    registry: SkillRegistry,
    channel_folder: Path,
) -> None:
    """
    title: Local folder channels should reject Git refs.
    parameters:
      registry:
        type: SkillRegistry
      channel_folder:
        type: Path
    """
    with pytest.raises(ValueError, match='ref is only supported'):
        registry.add_channel(str(channel_folder), ref='main')


def test_install_one_skill_from_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    title: One declared channel skill should install and load.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
    """
    registry.add_channel(str(channel_repo))
    installed = registry.install_skill('tm.ayurveda')

    assert installed == 'tm.ayurveda'
    state = registry._load_state()
    assert state.skills['tm.ayurveda'].manifest_path.endswith(
        'skills/ayurveda/skill.yaml'
    )

    skill = registry.load('tm.ayurveda')
    assert skill.metadata.name == 'tm.ayurveda'
    assert skill.metadata.version == '0.1.0'


def test_install_all_skills_from_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    title: Channel install should skip disabled skills by default.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
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
    title: Channel install can include disabled declared skills.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
    """
    registry.add_channel(str(channel_repo))

    installed = registry.install_channel('tm', include_disabled=True)

    assert installed == ['tm.ayurveda', 'tm.nutrition', 'tm.triage']


def test_update_one_skill_without_pulling_channel(
    registry: SkillRegistry,
    channel_repo: Path,
) -> None:
    """
    title: Updating a skill alone should not refresh the channel checkout.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
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
    title: Pulling a skill update should refresh its channel first.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
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
    title: Updating a channel should refresh installed skill metadata.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
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
    title: Removing one installed skill should leave the channel intact.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
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
    title: Removing a channel should remove its local state and skills.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
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
    title: StageRunner should register and run canonical channel skills.
    parameters:
      registry:
        type: SkillRegistry
      channel_repo:
        type: Path
    """
    registry.add_channel(str(channel_repo))
    registry.install_skill('tm.nutrition')

    runner = StageRunner(registry=registry)
    runner.register('tm.nutrition')
    ctx = runner.run(Stage.TREATMENT, PipelineContext(patient={}))

    assert runner.skills[0].metadata.name == 'tm.nutrition'
    assert ctx.extras['nutrition'] == 'fiber first'

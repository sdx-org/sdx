"""
title: Tests for dependency installation atomicity and defensive loading.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hiperhealth.pipeline import SkillRegistry
from hiperhealth.pipeline.registry import MissingDependencyError

from ._registry_test_utils import (
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


def _create_channel_with_deps(
    base_dir: Path,
    dependencies: list[str],
) -> Path:
    """
    title: Create a channel whose ayurveda skill declares dependencies.
    parameters:
      base_dir:
        type: Path
      dependencies:
        type: list[str]
    returns:
      type: Path
    """
    repo = create_channel_repo(base_dir)
    deps_yaml = '\n'.join(f'  - {dep}' for dep in dependencies)
    write_file(
        repo / 'skills' / 'ayurveda' / 'skill.yaml',
        f"""\
        api_version: 1
        name: ayurveda
        version: 0.1.0
        entry_point: skill:AyurvedaSkill
        stages:
          - diagnosis
          - treatment
        description: Ayurvedic reasoning support
        author: Test Org
        license: BSD-3-Clause
        homepage: https://example.com/traditional-medicine
        min_hiperhealth_version: ">=0.5.0"
        dependencies:
        {deps_yaml}
        """,
    )

    # Re-commit after editing the manifest.
    from ._registry_test_utils import commit_all

    commit_all(repo, 'add dependencies to ayurveda')
    return repo


class TestInstallSkillAtomicity:
    """
    title: Tests that state is only persisted after dependencies succeed.
    """

    def test_state_not_saved_when_pip_fails(
        self,
        registry: SkillRegistry,
        tmp_path: Path,
    ) -> None:
        """
        title: A pip failure should leave state unchanged.
        parameters:
          registry:
            type: SkillRegistry
          tmp_path:
            type: Path
        """
        repo = _create_channel_with_deps(
            tmp_path / 'repo', ['totally-fake-package>=99.0']
        )
        registry.add_channel(str(repo))

        with patch.object(
            registry,
            '_install_dependencies',
            side_effect=RuntimeError('pip failed'),
        ):
            with pytest.raises(RuntimeError, match='pip failed'):
                registry.install_skill('tm.ayurveda')

        state = registry._load_state()
        assert 'tm.ayurveda' not in state.skills

    def test_state_saved_when_pip_succeeds(
        self,
        registry: SkillRegistry,
        tmp_path: Path,
    ) -> None:
        """
        title: Successful pip should persist state normally.
        parameters:
          registry:
            type: SkillRegistry
          tmp_path:
            type: Path
        """
        repo = _create_channel_with_deps(
            tmp_path / 'repo', ['totally-fake-package>=99.0']
        )
        registry.add_channel(str(repo))

        with patch.object(
            registry, '_install_dependencies', return_value=None
        ):
            registry.install_skill('tm.ayurveda')

        state = registry._load_state()
        assert 'tm.ayurveda' in state.skills


class TestUpdateSkillAtomicity:
    """
    title: Tests that update_skill also installs deps before saving state.
    """

    def test_update_state_not_saved_when_pip_fails(
        self,
        registry: SkillRegistry,
        tmp_path: Path,
    ) -> None:
        """
        title: A pip failure during update should leave version unchanged.
        parameters:
          registry:
            type: SkillRegistry
          tmp_path:
            type: Path
        """
        repo = _create_channel_with_deps(
            tmp_path / 'repo', ['totally-fake-package>=99.0']
        )
        registry.add_channel(str(repo))

        with patch.object(
            registry, '_install_dependencies', return_value=None
        ):
            registry.install_skill('tm.ayurveda')

        original_state = registry._load_state()
        original_ts = original_state.skills['tm.ayurveda'].updated_at

        with patch.object(
            registry,
            '_install_dependencies',
            side_effect=RuntimeError('pip failed'),
        ):
            with pytest.raises(RuntimeError, match='pip failed'):
                registry.update_skill('tm.ayurveda')

        state_after = registry._load_state()
        assert state_after.skills['tm.ayurveda'].updated_at == original_ts


class TestVerifyDependencies:
    """
    title: Tests for the _verify_dependencies pre-flight check.
    """

    def test_empty_dependencies_passes(self) -> None:
        """
        title: An empty dependency list should never raise.
        """
        SkillRegistry._verify_dependencies('test.skill', [])

    def test_installed_package_passes(self) -> None:
        """
        title: A package known to be installed should pass verification.
        """
        # pytest is always available in a test environment.
        SkillRegistry._verify_dependencies('test.skill', ['pytest'])

    def test_missing_package_raises_error(self) -> None:
        """
        title: A missing package should raise MissingDependencyError.
        """
        with pytest.raises(MissingDependencyError) as exc_info:
            SkillRegistry._verify_dependencies(
                'test.skill', ['nonexistent-magic-package>=1.0']
            )
        assert exc_info.value.skill_id == 'test.skill'
        assert 'nonexistent-magic-package>=1.0' in exc_info.value.missing

    def test_partial_missing_reports_only_missing(self) -> None:
        """
        title: Only truly absent packages should appear in the error.
        """
        with pytest.raises(MissingDependencyError) as exc_info:
            SkillRegistry._verify_dependencies(
                'test.skill',
                ['pytest', 'does-not-exist-xyz'],
            )
        assert 'does-not-exist-xyz' in exc_info.value.missing
        assert 'pytest' not in exc_info.value.missing

    def test_version_specifiers_are_stripped(self) -> None:
        """
        title: Version specifiers should be stripped for the import check.
        """
        # pytest is installed; version spec should be ignored for the check.
        SkillRegistry._verify_dependencies('test.skill', ['pytest>=7.0,<99.0'])

    def test_hyphen_normalized_to_underscore(self) -> None:
        """
        title: Hyphens in package names should be normalized for importlib.
        """
        # pip-tools => pip_tools may or may not be installed,
        # but the point is to test normalization doesn't crash.
        with pytest.raises(MissingDependencyError):
            SkillRegistry._verify_dependencies(
                'test.skill', ['totally-fake-pkg']
            )


class TestDefensiveLoading:
    """
    title: Tests that load() raises MissingDependencyError for bad deps.
    """

    def test_load_raises_missing_dep_error(
        self,
        registry: SkillRegistry,
        tmp_path: Path,
    ) -> None:
        """
        title: load() should raise MissingDependencyError when deps missing.
        parameters:
          registry:
            type: SkillRegistry
          tmp_path:
            type: Path
        """
        repo = _create_channel_with_deps(
            tmp_path / 'repo', ['totally-fake-package>=99.0']
        )
        registry.add_channel(str(repo))

        # Force-install (bypass pip) to simulate a corrupted state.
        with patch.object(
            registry, '_install_dependencies', return_value=None
        ):
            registry.install_skill('tm.ayurveda')

        with pytest.raises(MissingDependencyError) as exc_info:
            registry.load('tm.ayurveda')

        assert exc_info.value.skill_id == 'tm.ayurveda'
        assert 'totally-fake-package>=99.0' in exc_info.value.missing

    def test_load_succeeds_with_no_deps(
        self,
        registry: SkillRegistry,
        tmp_path: Path,
    ) -> None:
        """
        title: Skills without dependencies should load normally.
        parameters:
          registry:
            type: SkillRegistry
          tmp_path:
            type: Path
        """
        repo = create_channel_repo(tmp_path / 'repo')
        registry.add_channel(str(repo))
        registry.install_skill('tm.ayurveda')

        skill = registry.load('tm.ayurveda')
        assert skill.metadata.name == 'tm.ayurveda'

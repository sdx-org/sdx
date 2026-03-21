"""
title: Tests for the SkillRegistry and skill installation workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hiperhealth.pipeline import (
    PipelineContext,
    SkillManifest,
    SkillRegistry,
    Stage,
    StageRunner,
    create_default_runner,
)

from tests.pipeline._registry_test_utils import create_channel_repo


@pytest.fixture
def tmp_registry(tmp_path: Path) -> SkillRegistry:
    """
    title: Provide a SkillRegistry backed by a temporary directory.
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
    title: Create a temporary channel repository fixture.
    parameters:
      tmp_path:
        type: Path
    returns:
      type: Path
    """
    return create_channel_repo(tmp_path)


class TestSkillManifest:
    """
    title: Tests for SkillManifest model validation.
    """

    def test_parse_minimal(self) -> None:
        """
        title: Minimal manifest should validate with defaults.
        """
        manifest = SkillManifest(
            name='test',
            version='1.0.0',
            entry_point='skill:TestSkill',
            stages=['screening'],
        )
        assert manifest.name == 'test'
        assert manifest.description == ''
        assert manifest.dependencies == []

    def test_parse_full(self) -> None:
        """
        title: Full manifest with all fields should validate.
        """
        manifest = SkillManifest(
            name='ayurveda',
            version='2.0.0',
            entry_point='skill:AyurvedaSkill',
            stages=['diagnosis', 'treatment'],
            description='Ayurvedic skill',
            author='Test Author',
            license='MIT',
            homepage='https://example.com',
            min_hiperhealth_version='0.4.0',
            dependencies=['some-package>=1.0'],
        )
        assert manifest.stages == ['diagnosis', 'treatment']
        assert manifest.dependencies == ['some-package>=1.0']


class TestSkillRegistryBuiltins:
    """
    title: Tests for built-in skill discovery via the registry.
    """

    def test_list_includes_builtins(self) -> None:
        """
        title: list_skills should include the three built-in skills.
        """
        registry = SkillRegistry()
        manifests = registry.list_skills()
        names = [m.name for m in manifests]

        assert 'hiperhealth.diagnostics' in names
        assert 'hiperhealth.extraction' in names
        assert 'hiperhealth.privacy' in names

    def test_load_builtin_privacy(self) -> None:
        """
        title: Loading the built-in privacy skill by name should work.
        """
        registry = SkillRegistry()
        skill = registry.load('hiperhealth.privacy')

        assert skill.metadata.name == 'hiperhealth.privacy'
        assert 'screening' in skill.metadata.stages

    def test_load_builtin_diagnostics(self) -> None:
        """
        title: Loading the built-in diagnostics skill by name should work.
        """
        registry = SkillRegistry()
        skill = registry.load('hiperhealth.diagnostics')

        assert skill.metadata.name == 'hiperhealth.diagnostics'
        assert 'diagnosis' in skill.metadata.stages

    def test_load_builtin_extraction(self) -> None:
        """
        title: Loading the built-in extraction skill by name should work.
        """
        registry = SkillRegistry()
        skill = registry.load('hiperhealth.extraction')

        assert skill.metadata.name == 'hiperhealth.extraction'
        assert 'intake' in skill.metadata.stages

    def test_load_nonexistent_raises(self) -> None:
        """
        title: Loading a non-existent skill should raise KeyError.
        """
        registry = SkillRegistry()
        with pytest.raises(KeyError, match='not found'):
            registry.load('nonexistent.skill')


class TestStageRunnerRegister:
    """
    title: Tests for StageRunner.register() integration with registry.
    """

    def test_register_builtin_skill(self) -> None:
        """
        title: register() should load a built-in skill by name.
        """
        runner = StageRunner()
        runner.register('hiperhealth.privacy')

        assert len(runner.skills) == 1
        assert runner.skills[0].metadata.name == 'hiperhealth.privacy'

    def test_register_with_index(self) -> None:
        """
        title: register() should respect the index parameter.
        """
        runner = StageRunner()
        runner.register('hiperhealth.diagnostics')
        runner.register('hiperhealth.privacy', index=0)

        names = [s.metadata.name for s in runner.skills]
        assert names == [
            'hiperhealth.privacy',
            'hiperhealth.diagnostics',
        ]

    def test_register_installed_skill(
        self,
        tmp_registry: SkillRegistry,
        channel_repo: Path,
    ) -> None:
        """
        title: register() should load an externally installed skill.
        parameters:
          tmp_registry:
            type: SkillRegistry
          channel_repo:
            type: Path
        """
        tmp_registry.add_channel(str(channel_repo))
        tmp_registry.install_skill('tm.ayurveda')

        runner = StageRunner(registry=tmp_registry)
        runner.register('tm.ayurveda')

        assert len(runner.skills) == 1
        assert runner.skills[0].metadata.name == 'tm.ayurveda'

    def test_register_and_run(
        self,
        tmp_registry: SkillRegistry,
        channel_repo: Path,
    ) -> None:
        """
        title: A registered skill should execute in the pipeline.
        parameters:
          tmp_registry:
            type: SkillRegistry
          channel_repo:
            type: Path
        """
        tmp_registry.add_channel(str(channel_repo))
        tmp_registry.install_skill('tm.ayurveda')

        runner = StageRunner(registry=tmp_registry)
        runner.register('tm.ayurveda')

        ctx = PipelineContext(patient={'name': 'Bob'})
        ctx = runner.run(Stage.TREATMENT, ctx)

        assert ctx.extras['ayurveda'] == 'warm herbs'


class TestCreateDefaultRunnerWithRegistry:
    """
    title: Tests for create_default_runner using the registry.
    """

    def test_creates_runner_with_builtin_skills(self) -> None:
        """
        title: create_default_runner should load built-in skills via registry.
        """
        runner = create_default_runner()
        names = [s.metadata.name for s in runner.skills]

        assert 'hiperhealth.privacy' in names
        assert 'hiperhealth.extraction' in names
        assert 'hiperhealth.diagnostics' in names

    def test_privacy_runs_first(self) -> None:
        """
        title: Privacy skill should be first in registration order.
        """
        runner = create_default_runner()
        names = [s.metadata.name for s in runner.skills]

        assert names[0] == 'hiperhealth.privacy'

    def test_registration_order(self) -> None:
        """
        title: Built-in skills should be in expected order.
        """
        runner = create_default_runner()
        names = [s.metadata.name for s in runner.skills]

        assert names == [
            'hiperhealth.privacy',
            'hiperhealth.extraction',
            'hiperhealth.diagnostics',
        ]


class TestReadManifest:
    """
    title: Tests for reading skill.yaml manifest files.
    """

    def test_read_builtin_manifests(self) -> None:
        """
        title: All built-in skill directories should have valid manifests.
        """
        registry = SkillRegistry()
        manifests = registry.list_skills()

        for manifest in manifests:
            assert manifest.name
            assert manifest.version
            assert manifest.entry_point
            assert len(manifest.stages) > 0

    def test_manifest_stages_match_skill(self) -> None:
        """
        title: Manifest stages should match the skill's metadata stages.
        """
        registry = SkillRegistry()

        for name in [
            'hiperhealth.privacy',
            'hiperhealth.extraction',
            'hiperhealth.diagnostics',
        ]:
            manifests = registry.list_skills()
            manifest = next(m for m in manifests if m.name == name)
            skill = registry.load(name)

            assert set(manifest.stages) == set(skill.metadata.stages)

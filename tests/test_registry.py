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
    return SkillRegistry(registry_dir=tmp_path / 'skills')


@pytest.fixture
def sample_skill_dir(tmp_path: Path) -> Path:
    """
    title: Create a minimal skill project directory for testing.
    parameters:
      tmp_path:
        type: Path
    returns:
      type: Path
    """
    skill_dir = tmp_path / 'sample_skill'
    skill_dir.mkdir()

    yaml_content = (
        'name: test.greeting\n'
        'version: 1.0.0\n'
        'entry_point: "skill:GreetingSkill"\n'
        'stages:\n'
        '  - screening\n'
        'description: A test greeting skill.\n'
    )
    (skill_dir / 'hiperhealth.yaml').write_text(yaml_content)

    skill_code = (
        'from hiperhealth.pipeline import BaseSkill, SkillMetadata\n'
        'from hiperhealth.pipeline.context import PipelineContext\n'
        '\n'
        '\n'
        'class GreetingSkill(BaseSkill):\n'
        '    def __init__(self):\n'
        '        super().__init__(\n'
        '            SkillMetadata(\n'
        '                name="test.greeting",\n'
        '                version="1.0.0",\n'
        '                stages=("screening",),\n'
        '                description="A test greeting skill.",\n'
        '            )\n'
        '        )\n'
        '\n'
        '    def execute(self, stage, ctx):\n'
        '        name = ctx.patient.get("name", "Patient")\n'
        '        ctx.extras["greeting"] = f"Welcome, {name}!"\n'
        '        return ctx\n'
    )
    (skill_dir / 'skill.py').write_text(skill_code)

    return skill_dir


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


class TestSkillRegistryInstall:
    """
    title: Tests for installing skills from local paths.
    """

    def test_install_from_path(
        self,
        tmp_registry: SkillRegistry,
        sample_skill_dir: Path,
    ) -> None:
        """
        title: Installing a skill from a local path should copy it.
        parameters:
          tmp_registry:
            type: SkillRegistry
          sample_skill_dir:
            type: Path
        """
        name = tmp_registry.install(str(sample_skill_dir))

        assert name == 'test.greeting'
        # Skill directory should exist in registry
        assert (tmp_registry.registry_dir / 'test.greeting').is_dir()
        # manifest.json should exist
        assert (tmp_registry.registry_dir / 'manifest.json').exists()

    def test_installed_skill_appears_in_list(
        self,
        tmp_registry: SkillRegistry,
        sample_skill_dir: Path,
    ) -> None:
        """
        title: Installed skills should appear in list_skills.
        parameters:
          tmp_registry:
            type: SkillRegistry
          sample_skill_dir:
            type: Path
        """
        tmp_registry.install(str(sample_skill_dir))
        manifests = tmp_registry.list_skills()
        names = [m.name for m in manifests]

        assert 'test.greeting' in names

    def test_load_installed_skill(
        self,
        tmp_registry: SkillRegistry,
        sample_skill_dir: Path,
    ) -> None:
        """
        title: Loading an installed skill should return an instance.
        parameters:
          tmp_registry:
            type: SkillRegistry
          sample_skill_dir:
            type: Path
        """
        tmp_registry.install(str(sample_skill_dir))
        skill = tmp_registry.load('test.greeting')

        assert skill.metadata.name == 'test.greeting'
        assert skill.metadata.version == '1.0.0'

    def test_installed_skill_executes(
        self,
        tmp_registry: SkillRegistry,
        sample_skill_dir: Path,
    ) -> None:
        """
        title: An installed skill should execute correctly in a runner.
        parameters:
          tmp_registry:
            type: SkillRegistry
          sample_skill_dir:
            type: Path
        """
        tmp_registry.install(str(sample_skill_dir))
        skill = tmp_registry.load('test.greeting')

        runner = StageRunner(skills=[skill])
        ctx = PipelineContext(patient={'name': 'Alice'})
        ctx = runner.run(Stage.SCREENING, ctx)

        assert ctx.extras['greeting'] == 'Welcome, Alice!'

    def test_uninstall_removes_skill(
        self,
        tmp_registry: SkillRegistry,
        sample_skill_dir: Path,
    ) -> None:
        """
        title: Uninstalling a skill should remove it from the registry.
        parameters:
          tmp_registry:
            type: SkillRegistry
          sample_skill_dir:
            type: Path
        """
        tmp_registry.install(str(sample_skill_dir))
        tmp_registry.uninstall('test.greeting')

        assert not (tmp_registry.registry_dir / 'test.greeting').exists()
        with pytest.raises(KeyError):
            tmp_registry.load('test.greeting')

    def test_reinstall_overwrites(
        self,
        tmp_registry: SkillRegistry,
        sample_skill_dir: Path,
    ) -> None:
        """
        title: Re-installing a skill should overwrite the previous version.
        parameters:
          tmp_registry:
            type: SkillRegistry
          sample_skill_dir:
            type: Path
        """
        tmp_registry.install(str(sample_skill_dir))
        tmp_registry.install(str(sample_skill_dir))

        # Should still work fine
        skill = tmp_registry.load('test.greeting')
        assert skill.metadata.name == 'test.greeting'

    def test_install_invalid_source_raises(
        self,
        tmp_registry: SkillRegistry,
    ) -> None:
        """
        title: Installing from an invalid source should raise ValueError.
        parameters:
          tmp_registry:
            type: SkillRegistry
        """
        with pytest.raises(ValueError, match='Cannot install'):
            tmp_registry.install('not-a-path-or-url')

    def test_install_missing_yaml_raises(
        self,
        tmp_registry: SkillRegistry,
        tmp_path: Path,
    ) -> None:
        """
        title: Installing a dir without hiperhealth.yaml should raise.
        parameters:
          tmp_registry:
            type: SkillRegistry
          tmp_path:
            type: Path
        """
        empty_dir = tmp_path / 'empty_skill'
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match=r'hiperhealth\.yaml'):
            tmp_registry.install(str(empty_dir))


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
        sample_skill_dir: Path,
    ) -> None:
        """
        title: register() should load an externally installed skill.
        parameters:
          tmp_registry:
            type: SkillRegistry
          sample_skill_dir:
            type: Path
        """
        tmp_registry.install(str(sample_skill_dir))

        runner = StageRunner(registry=tmp_registry)
        runner.register('test.greeting')

        assert len(runner.skills) == 1
        assert runner.skills[0].metadata.name == 'test.greeting'

    def test_register_and_run(
        self,
        tmp_registry: SkillRegistry,
        sample_skill_dir: Path,
    ) -> None:
        """
        title: A registered skill should execute in the pipeline.
        parameters:
          tmp_registry:
            type: SkillRegistry
          sample_skill_dir:
            type: Path
        """
        tmp_registry.install(str(sample_skill_dir))

        runner = StageRunner(registry=tmp_registry)
        runner.register('test.greeting')

        ctx = PipelineContext(patient={'name': 'Bob'})
        ctx = runner.run(Stage.SCREENING, ctx)

        assert ctx.extras['greeting'] == 'Welcome, Bob!'


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
    title: Tests for reading hiperhealth.yaml manifest files.
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

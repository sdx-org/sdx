"""
title: Helpers for registry channel and CLI tests.
"""

from __future__ import annotations

import re
import subprocess
import textwrap

from pathlib import Path


def write_file(path: Path, content: str) -> None:
    """
    title: Write a text file, creating parent directories as needed
    parameters:
      path:
        type: Path
        description: Output file path.
      content:
        type: str
        description: File content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(content).strip() + '\n',
        encoding='utf-8',
    )


def run_git(repo: Path, *args: str) -> str:
    """
    title: Run a git command inside a repository
    parameters:
      repo:
        type: Path
        description: Repository directory.
      args:
        type: str
        description: Git arguments.
        variadic: positional
    returns:
      type: str
      description: Command stdout.
    """
    completed = subprocess.run(
        ['git', *args],
        check=True,
        capture_output=True,
        cwd=repo,
        text=True,
    )
    return completed.stdout.strip()


def init_git_repo(repo: Path) -> None:
    """
    title: Initialize a local git repository for tests
    parameters:
      repo:
        type: Path
        description: Repository directory.
    """
    repo.mkdir(parents=True, exist_ok=True)
    run_git(repo, 'init')
    run_git(repo, 'config', 'user.name', 'Registry Tests')
    run_git(repo, 'config', 'user.email', 'registry-tests@example.com')
    run_git(repo, 'checkout', '-b', 'main')


def commit_all(repo: Path, message: str) -> None:
    """
    title: Commit all current repository changes
    parameters:
      repo:
        type: Path
        description: Repository directory.
      message:
        type: str
        description: Commit message.
    """
    run_git(repo, 'add', '.')
    run_git(repo, 'commit', '-m', message)


def create_channel_repo(base_dir: Path, *, use_git: bool = True) -> Path:
    """
    title: Create a channel repository fixture
    parameters:
      base_dir:
        type: Path
        description: Directory where the repository should be created.
      use_git:
        type: bool
        description: Whether to initialize the fixture as a git repository.
    returns:
      type: Path
      description: Repository path.
    """
    repo = base_dir / 'traditional-medicine'
    if use_git:
        init_git_repo(repo)
    else:
        repo.mkdir(parents=True, exist_ok=True)

    write_file(
        repo / 'skills-channel.yaml',
        """
        api_version: 1
        channel:
          name: traditional-medicine
          display_name: Traditional Medicine
          default_alias: tm
          version: 0.1.0
          description: Complementary and traditional medicine skills
          homepage: https://example.com/traditional-medicine
          license: BSD-3-Clause
          min_hiperhealth_version: ">=0.5.0"

        skills:
          - name: ayurveda
            enabled: true
            tags: [traditional-medicine, treatment]

          - name: nutrition
            enabled: true
            tags: [nutrition]

          - name: triage
            enabled: false
            tags: [screening]
        """,
    )
    write_file(repo / 'README.md', '# Traditional Medicine')
    write_file(repo / 'infra' / 'README.md', 'infra')

    write_file(
        repo / 'skills' / 'ayurveda' / 'skill.yaml',
        """
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
        dependencies: []
        """,
    )
    write_file(
        repo / 'skills' / 'ayurveda' / 'skill.py',
        """
        from hiperhealth.pipeline import BaseSkill, SkillMetadata


        class AyurvedaSkill(BaseSkill):
            def __init__(self) -> None:
                super().__init__(
                    SkillMetadata(
                        name='ayurveda',
                        version='0.1.0',
                        stages=('diagnosis', 'treatment'),
                        description='Ayurvedic reasoning support',
                    )
                )

            def execute(self, stage, ctx):
                ctx.extras['ayurveda'] = 'warm herbs'
                return ctx
        """,
    )

    write_file(
        repo / 'skills' / 'nutrition' / 'skill.yaml',
        """
        api_version: 1
        name: nutrition
        version: 0.1.0
        entry_point: package_impl.skill:NutritionSkill
        stages:
          - treatment
        description: Nutrition guidance support
        author: Test Org
        license: BSD-3-Clause
        homepage: https://example.com/traditional-medicine
        min_hiperhealth_version: ">=0.5.0"
        dependencies: []
        """,
    )
    write_file(
        repo / 'skills' / 'nutrition' / 'package_impl' / '__init__.py',
        """
        from .helpers import nutrition_message

        __all__ = ['nutrition_message']
        """,
    )
    write_file(
        repo / 'skills' / 'nutrition' / 'package_impl' / 'helpers.py',
        """
        def nutrition_message() -> str:
            return 'fiber first'
        """,
    )
    write_file(
        repo / 'skills' / 'nutrition' / 'package_impl' / 'skill.py',
        """
        from hiperhealth.pipeline import BaseSkill, SkillMetadata

        from .helpers import nutrition_message


        class NutritionSkill(BaseSkill):
            def __init__(self) -> None:
                super().__init__(
                    SkillMetadata(
                        name='nutrition',
                        version='0.1.0',
                        stages=('treatment',),
                        description='Nutrition guidance support',
                    )
                )

            def execute(self, stage, ctx):
                ctx.extras['nutrition'] = nutrition_message()
                return ctx
        """,
    )

    write_file(
        repo / 'skills' / 'triage' / 'skill.yaml',
        """
        api_version: 1
        name: triage
        version: 0.1.0
        entry_point: skill:TriageSkill
        stages:
          - screening
        description: Basic triage support
        author: Test Org
        license: BSD-3-Clause
        homepage: https://example.com/traditional-medicine
        min_hiperhealth_version: ">=0.5.0"
        dependencies: []
        """,
    )
    write_file(
        repo / 'skills' / 'triage' / 'skill.py',
        """
        from hiperhealth.pipeline import BaseSkill, SkillMetadata


        class TriageSkill(BaseSkill):
            def __init__(self) -> None:
                super().__init__(
                    SkillMetadata(
                        name='triage',
                        version='0.1.0',
                        stages=('screening',),
                        description='Basic triage support',
                    )
                )

            def execute(self, stage, ctx):
                ctx.extras['triage'] = 'screened'
                return ctx
        """,
    )

    if use_git:
        commit_all(repo, 'initial channel')
    return repo


def bump_channel_skill_version(
    repo: Path,
    skill_name: str,
    new_version: str,
) -> None:
    """
    title: Update a channel skill version in both manifest and Python code
    parameters:
      repo:
        type: Path
        description: Channel repository path.
      skill_name:
        type: str
        description: Channel-local skill name.
      new_version:
        type: str
        description: Replacement version string.
    """
    manifest_path = repo / 'skills' / skill_name / 'skill.yaml'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    match = re.search(r'^version:\s*(.+)$', manifest_text, re.MULTILINE)
    assert match is not None
    old_version = match.group(1).strip()
    manifest_path.write_text(
        manifest_text.replace(
            f'version: {old_version}',
            f'version: {new_version}',
            1,
        ),
        encoding='utf-8',
    )

    module_candidates = [
        repo / 'skills' / skill_name / 'skill.py',
        repo / 'skills' / skill_name / 'package_impl' / 'skill.py',
    ]
    for module_path in module_candidates:
        if not module_path.exists():
            continue
        module_text = module_path.read_text(encoding='utf-8')
        module_path.write_text(
            module_text.replace(
                f"version='{old_version}'",
                f"version='{new_version}'",
                1,
            ),
            encoding='utf-8',
        )
        break

    commit_all(repo, f'update {skill_name} to {new_version}')

"""
title: SkillRegistry — manages skill installation and loading.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hiperhealth.pipeline.skill import BaseSkill


class SkillManifest(BaseModel):
    """
    title: Metadata parsed from a skill's hiperhealth.yaml file.
    attributes:
      name:
        type: str
      version:
        type: str
      entry_point:
        type: str
      stages:
        type: list[str]
      description:
        type: str
      author:
        type: str
      license:
        type: str
      homepage:
        type: str
      min_hiperhealth_version:
        type: str
      dependencies:
        type: list[str]
    """

    name: str
    version: str
    entry_point: str
    stages: list[str]
    description: str = ''
    author: str = ''
    license: str = ''
    homepage: str = ''
    min_hiperhealth_version: str = ''
    dependencies: list[str] = Field(default_factory=list)


class InstalledSkillRecord(BaseModel):
    """
    title: Tracks the origin of an installed skill.
    attributes:
      source:
        type: str
      installed_at:
        type: str
      version:
        type: str
    """

    source: str
    installed_at: str
    version: str


class RegistryIndex(BaseModel):
    """
    title: Persisted index of externally installed skills.
    attributes:
      skills:
        type: dict[str, InstalledSkillRecord]
    """

    skills: dict[str, InstalledSkillRecord] = Field(default_factory=dict)


def _parse_yaml(path: Path) -> dict[str, Any]:
    """
    title: Parse a YAML file into a dict.
    parameters:
      path:
        type: Path
    returns:
      type: dict[str, Any]
    """
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_class_from_path(
    skill_dir: Path,
    entry_point: str,
) -> type:
    """
    title: Dynamically load a skill class from a directory and entry point.
    summary: |-
      The entry_point format is ``module:ClassName``. The module is
      loaded from ``skill_dir/module.py``.
    parameters:
      skill_dir:
        type: Path
      entry_point:
        type: str
    returns:
      type: type
    """
    module_name, class_name = entry_point.split(':')
    module_path = skill_dir / f'{module_name}.py'

    spec = importlib.util.spec_from_file_location(
        f'_hiperhealth_skill_{module_name}',
        module_path,
    )
    if spec is None or spec.loader is None:
        msg = f'Cannot load module from {module_path}'
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return getattr(module, class_name)  # type: ignore[no-any-return]


def _load_class_from_package(
    package_base: str,
    entry_point: str,
) -> type:
    """
    title: Load a skill class from an installed Python package.
    summary: |-
      For built-in skills, the package_base is something like
      ``hiperhealth.skills.diagnostics`` and entry_point is
      ``core:DiagnosticsSkill``.
    parameters:
      package_base:
        type: str
      entry_point:
        type: str
    returns:
      type: type
    """
    module_name, class_name = entry_point.split(':')
    full_module = f'{package_base}.{module_name}'
    module = importlib.import_module(full_module)
    return getattr(module, class_name)  # type: ignore[no-any-return]


class SkillRegistry:
    """
    title: Manages skill installation, discovery, and loading.
    summary: |-
      Skills can be installed from local paths or Git URLs.
      Built-in skills are auto-discovered from the hiperhealth package.
      Use ``load()`` to instantiate a skill by name.
    attributes:
      _registry_dir:
        description: Value for _registry_dir.
      _builtin_dir:
        description: Value for _builtin_dir.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self._registry_dir = (
            registry_dir or Path.home() / '.hiperhealth' / 'skills'
        )
        self._builtin_dir = Path(__file__).resolve().parent.parent / 'skills'

    @property
    def registry_dir(self) -> Path:
        """
        title: Return the external skills registry directory.
        returns:
          type: Path
        """
        return self._registry_dir

    def _ensure_registry_dir(self) -> None:
        """
        title: Create the registry directory if it does not exist.
        """
        self._registry_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> RegistryIndex:
        """
        title: Load the registry manifest.json index.
        returns:
          type: RegistryIndex
        """
        manifest_path = self._registry_dir / 'manifest.json'
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            return RegistryIndex.model_validate(data)
        return RegistryIndex()

    def _save_index(self, index: RegistryIndex) -> None:
        """
        title: Persist the registry index to manifest.json.
        parameters:
          index:
            type: RegistryIndex
        """
        self._ensure_registry_dir()
        manifest_path = self._registry_dir / 'manifest.json'
        manifest_path.write_text(index.model_dump_json(indent=2) + '\n')

    def _read_manifest(self, skill_dir: Path) -> SkillManifest:
        """
        title: Read and validate hiperhealth.yaml from a skill directory.
        parameters:
          skill_dir:
            type: Path
        returns:
          type: SkillManifest
        """
        yaml_path = skill_dir / 'hiperhealth.yaml'
        if not yaml_path.exists():
            msg = (
                f'No hiperhealth.yaml found in {skill_dir}. '
                f'Every skill project must include a hiperhealth.yaml '
                f'metadata file.'
            )
            raise FileNotFoundError(msg)
        data = _parse_yaml(yaml_path)
        return SkillManifest.model_validate(data)

    def install(self, source: str) -> str:
        """
        title: Install a skill from a local path or Git URL.
        summary: |-
          Copies the skill folder into the internal registry directory.
          For Git URLs, the repository is cloned first.
          Returns the installed skill name.
        parameters:
          source:
            type: str
        returns:
          type: str
        """
        source_path = Path(source)

        if source_path.is_dir():
            return self._install_from_path(source_path, source)

        if source.startswith(('https://', 'git@', 'http://')):
            return self._install_from_git(source)

        msg = (
            f'Cannot install from {source!r}. '
            f'Provide a local directory path or a Git URL.'
        )
        raise ValueError(msg)

    def _install_from_path(self, source_path: Path, source_str: str) -> str:
        """
        title: Install a skill from a local directory.
        parameters:
          source_path:
            type: Path
          source_str:
            type: str
        returns:
          type: str
        """
        manifest = self._read_manifest(source_path)
        self._ensure_registry_dir()

        target = self._registry_dir / manifest.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_path, target)

        index = self._load_index()
        index.skills[manifest.name] = InstalledSkillRecord(
            source=source_str,
            installed_at=datetime.now(timezone.utc).isoformat(),
            version=manifest.version,
        )
        self._save_index(index)

        if manifest.dependencies:
            self._install_dependencies(manifest.dependencies)

        return manifest.name

    def _install_from_git(self, url: str) -> str:
        """
        title: Install a skill from a Git repository URL.
        parameters:
          url:
            type: str
        returns:
          type: str
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / 'repo'
            subprocess.run(
                ['git', 'clone', '--depth', '1', url, str(tmp_path)],
                check=True,
                capture_output=True,
            )
            return self._install_from_path(tmp_path, url)

    def _install_dependencies(self, deps: list[str]) -> None:
        """
        title: Install pip dependencies for a skill.
        parameters:
          deps:
            type: list[str]
        """
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', *deps],
            check=True,
            capture_output=True,
        )

    def uninstall(self, name: str) -> None:
        """
        title: Remove an installed skill from the registry.
        parameters:
          name:
            type: str
        """
        target = self._registry_dir / name
        if target.exists():
            shutil.rmtree(target)

        index = self._load_index()
        index.skills.pop(name, None)
        self._save_index(index)

    def list_skills(self) -> list[SkillManifest]:
        """
        title: List all available skills (built-in and installed).
        returns:
          type: list[SkillManifest]
        """
        manifests: list[SkillManifest] = []

        # Built-in skills
        if self._builtin_dir.is_dir():
            for child in sorted(self._builtin_dir.iterdir()):
                yaml_path = child / 'hiperhealth.yaml'
                if yaml_path.exists():
                    manifests.append(self._read_manifest(child))

        # Externally installed skills
        if self._registry_dir.is_dir():
            for child in sorted(self._registry_dir.iterdir()):
                yaml_path = child / 'hiperhealth.yaml'
                if yaml_path.exists():
                    manifests.append(self._read_manifest(child))

        return manifests

    def load(self, name: str) -> BaseSkill:
        """
        title: Instantiate a skill by name.
        summary: |-
          Searches built-in skills first, then the external registry.
          Returns an instance of the skill class.
        parameters:
          name:
            type: str
        returns:
          type: BaseSkill
        """
        # Check built-in skills first
        if self._builtin_dir.is_dir():
            for child in self._builtin_dir.iterdir():
                yaml_path = child / 'hiperhealth.yaml'
                if not yaml_path.exists():
                    continue
                manifest = self._read_manifest(child)
                if manifest.name == name:
                    package_base = f'hiperhealth.skills.{child.name}'
                    cls = _load_class_from_package(
                        package_base, manifest.entry_point
                    )
                    skill: BaseSkill = cls()
                    return skill

        # Check external registry
        external_dir = self._registry_dir / name
        if external_dir.is_dir():
            manifest = self._read_manifest(external_dir)
            cls = _load_class_from_path(external_dir, manifest.entry_point)
            skill = cls()
            return skill

        msg = (
            f'Skill {name!r} not found. '
            f'Use list_skills() to see available skills, or '
            f'install() to add new ones.'
        )
        raise KeyError(msg)

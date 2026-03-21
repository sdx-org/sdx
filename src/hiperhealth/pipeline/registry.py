"""
title: Channel-aware skill registry for built-in and channel skills.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from hiperhealth.pipeline.skill import BaseSkill, SkillMetadata

BUILTIN_CHANNEL = 'hiperhealth'
_LOCAL_NAME_PATTERN = re.compile(r'^[A-Za-z0-9_-]+$')


class SkillManifest(BaseModel):
    api_version: int = 1
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


class ChannelMetadata(BaseModel):
    name: str
    display_name: str = ''
    default_alias: str | None = None
    version: str = ''
    description: str = ''
    homepage: str = ''
    license: str = ''
    min_hiperhealth_version: str = ''


class DeclaredSkill(BaseModel):
    name: str
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)


class ChannelManifest(BaseModel):
    api_version: int = 1
    channel: ChannelMetadata
    skills: list[DeclaredSkill] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_skill_names(self) -> ChannelManifest:
        """
        title: Ensure declared channel skill names are unique.
        returns:
          type: ChannelManifest
        """
        names = [skill.name for skill in self.skills]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            joined = ', '.join(duplicates)
            msg = (
                'Duplicate skill names declared in '
                f'skills-channel.yaml: {joined}.'
            )
            raise ValueError(msg)
        return self


class ChannelRecord(BaseModel):
    local_name: str
    remote_name: str
    provider: str
    source: str
    ref: str | None = None
    commit: str = ''
    registered_at: str
    updated_at: str
    available_skills: list[str] = Field(default_factory=list)


class AvailableSkillRecord(BaseModel):
    channel: str
    name: str
    canonical_id: str
    path: str
    manifest_path: str
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)


class InstalledSkillRecord(BaseModel):
    id: str
    channel: str | None = None
    skill_name: str
    manifest_path: str
    installed_at: str
    updated_at: str
    version: str
    source_commit: str = ''
    enabled: bool = True


class RegistryState(BaseModel):
    channels: dict[str, ChannelRecord] = Field(default_factory=dict)
    skills: dict[str, InstalledSkillRecord] = Field(default_factory=dict)


class SkillSummary(SkillManifest):
    channel: str | None = None
    skill_name: str
    canonical_id: str
    manifest_path: str
    installed: bool = False
    enabled: bool = True
    builtin: bool = False
    tags: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _ResolvedChannelSkill:
    available: AvailableSkillRecord
    manifest: SkillManifest
    manifest_path: Path


def _parse_yaml(path: Path) -> dict[str, Any]:
    """
    title: Parse a YAML file into a plain Python dictionary.
    parameters:
      path:
        type: Path
    returns:
      type: dict[str, Any]
    """
    import yaml

    with path.open(encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def _utcnow() -> str:
    """
    title: Return the current UTC timestamp in ISO 8601 format.
    returns:
      type: str
    """
    return datetime.now(timezone.utc).isoformat()


def _canonical_skill_id(local_name: str, skill_name: str) -> str:
    """
    title: Build a canonical skill id from channel and skill names.
    parameters:
      local_name:
        type: str
      skill_name:
        type: str
    returns:
      type: str
    """
    return f'{local_name}.{skill_name}'


def _split_entry_point(entry_point: str) -> tuple[str, str]:
    """
    title: Split a manifest entry point into module and class names.
    parameters:
      entry_point:
        type: str
    returns:
      type: tuple[str, str]
    """
    module_name, separator, class_name = entry_point.partition(':')
    if not separator or not module_name or not class_name:
        msg = (
            f'Invalid entry point {entry_point!r}. Expected '
            '"module:ClassName".'
        )
        raise ValueError(msg)
    return module_name, class_name


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    """
    title: Temporarily prepend a directory to sys.path.
    parameters:
      path:
        type: Path
    returns:
      type: Iterator[None]
    """
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if value in sys.path:
            sys.path.remove(value)


def _load_class_from_directory(skill_dir: Path, entry_point: str) -> type[Any]:
    """
    title: Load a skill class from a channel skill directory.
    parameters:
      skill_dir:
        type: Path
      entry_point:
        type: str
    returns:
      type: type[Any]
    """
    module_name, class_name = _split_entry_point(entry_point)
    root_module = module_name.split('.')[0]
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == root_module or name.startswith(f'{root_module}.'):
            del sys.modules[name]

    with _prepend_sys_path(skill_dir):
        module = importlib.import_module(module_name)

    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        msg = f'Entry point {entry_point!r} does not resolve to a class.'
        raise TypeError(msg)
    return cls


def _load_class_from_package(package_base: str, entry_point: str) -> type[Any]:
    """
    title: Load a built-in skill class from a package path.
    parameters:
      package_base:
        type: str
      entry_point:
        type: str
    returns:
      type: type[Any]
    """
    module_name, class_name = _split_entry_point(entry_point)
    module = importlib.import_module(f'{package_base}.{module_name}')
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        msg = f'Entry point {entry_point!r} does not resolve to a class.'
        raise TypeError(msg)
    return cls


class SkillRegistry:
    def __init__(self, registry_dir: Path | None = None) -> None:
        """
        title: Initialize the channel-aware skill registry.
        parameters:
          registry_dir:
            type: Path | None
        """
        self._registry_dir = (
            registry_dir
            or Path.home() / '.hiperhealth' / 'artifacts' / 'skills'
        )
        if (
            self._registry_dir.name == 'skills'
            and self._registry_dir.parent.name == 'artifacts'
        ):
            self._root_dir = self._registry_dir.parent.parent
        else:
            self._root_dir = self._registry_dir.parent
        self._state_dir = self._root_dir / 'registry'
        self._channels_dir = self._root_dir / 'channels'
        self._builtin_dir = Path(__file__).resolve().parent.parent / 'skills'

    @property
    def registry_dir(self) -> Path:
        """
        title: Return the directory used for installed skill artifacts.
        returns:
          type: Path
        """
        return self._registry_dir

    @property
    def root_dir(self) -> Path:
        """
        title: Return the root directory for registry state and channels.
        returns:
          type: Path
        """
        return self._root_dir

    def _ensure_storage_dirs(self) -> None:
        """
        title: Create the on-disk directories required by the registry.
        """
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._channels_dir.mkdir(parents=True, exist_ok=True)
        self._registry_dir.mkdir(parents=True, exist_ok=True)

    def _run_command(self, args: list[str], cwd: Path | None = None) -> str:
        """
        title: Run a subprocess command and return trimmed stdout.
        parameters:
          args:
            type: list[str]
          cwd:
            type: Path | None
        returns:
          type: str
        """
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return completed.stdout.strip()

    def _state_path(self) -> Path:
        """
        title: Return the registry state file path.
        returns:
          type: Path
        """
        return self._state_dir / 'state.json'

    def _channels_index_path(self) -> Path:
        """
        title: Return the legacy channels index file path.
        returns:
          type: Path
        """
        return self._state_dir / 'channels.json'

    def _skills_index_path(self) -> Path:
        """
        title: Return the legacy installed-skills index file path.
        returns:
          type: Path
        """
        return self._state_dir / 'skills.json'

    def _channel_dir(self, local_name: str) -> Path:
        """
        title: Return the storage directory for a registered channel.
        parameters:
          local_name:
            type: str
        returns:
          type: Path
        """
        return self._channels_dir / local_name

    def _channel_repo_dir(self, local_name: str) -> Path:
        """
        title: Return the local checkout directory for a channel.
        parameters:
          local_name:
            type: str
        returns:
          type: Path
        """
        return self._channel_dir(local_name) / 'repo'

    def _channel_manifest_copy_path(self, local_name: str) -> Path:
        """
        title: Return the cached channel manifest path.
        parameters:
          local_name:
            type: str
        returns:
          type: Path
        """
        return self._channel_dir(local_name) / 'skills-channel.yaml'

    def _channel_record_path(self, local_name: str) -> Path:
        """
        title: Return the per-channel metadata record path.
        parameters:
          local_name:
            type: str
        returns:
          type: Path
        """
        return self._channel_dir(local_name) / 'channel.json'

    def _skill_dir(self, repo_dir: Path, skill_name: str) -> Path:
        """
        title: Return the expected directory for a channel skill.
        parameters:
          repo_dir:
            type: Path
          skill_name:
            type: str
        returns:
          type: Path
        """
        return repo_dir / 'skills' / skill_name

    def _skill_manifest_path(self, repo_dir: Path, skill_name: str) -> Path:
        """
        title: Return the expected manifest path for a channel skill.
        parameters:
          repo_dir:
            type: Path
          skill_name:
            type: str
        returns:
          type: Path
        """
        return self._skill_dir(repo_dir, skill_name) / 'skill.yaml'

    def _load_state(self) -> RegistryState:
        """
        title: Load registry state from disk.
        returns:
          type: RegistryState
        """
        state_path = self._state_path()
        if state_path.exists():
            return RegistryState.model_validate_json(
                state_path.read_text(encoding='utf-8')
            )

        channels_path = self._channels_index_path()
        skills_path = self._skills_index_path()
        data: dict[str, Any] = {
            'channels': {},
            'skills': {},
        }
        if channels_path.exists():
            data['channels'] = json.loads(
                channels_path.read_text(encoding='utf-8')
            )
        if skills_path.exists():
            data['skills'] = json.loads(
                skills_path.read_text(encoding='utf-8')
            )
        return RegistryState.model_validate(data)

    def _save_state(self, state: RegistryState) -> None:
        """
        title: Persist registry state and compatibility index files.
        parameters:
          state:
            type: RegistryState
        """
        self._ensure_storage_dirs()
        state_data = state.model_dump(mode='json')
        self._state_path().write_text(
            json.dumps(state_data, indent=2) + '\n',
            encoding='utf-8',
        )
        self._channels_index_path().write_text(
            json.dumps(state_data['channels'], indent=2) + '\n',
            encoding='utf-8',
        )
        self._skills_index_path().write_text(
            json.dumps(state_data['skills'], indent=2) + '\n',
            encoding='utf-8',
        )

    def _save_channel_metadata(
        self,
        local_name: str,
        record: ChannelRecord,
        repo_dir: Path,
    ) -> None:
        """
        title: Persist a channel record and cached manifest copy.
        parameters:
          local_name:
            type: str
          record:
            type: ChannelRecord
          repo_dir:
            type: Path
        """
        channel_dir = self._channel_dir(local_name)
        channel_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            repo_dir / 'skills-channel.yaml',
            self._channel_manifest_copy_path(local_name),
        )
        self._channel_record_path(local_name).write_text(
            record.model_dump_json(indent=2) + '\n',
            encoding='utf-8',
        )

    def _read_skill_manifest_file(self, manifest_path: Path) -> SkillManifest:
        """
        title: Read and validate a per-skill manifest file.
        parameters:
          manifest_path:
            type: Path
        returns:
          type: SkillManifest
        """
        if not manifest_path.exists():
            msg = (
                f'No skill.yaml found at {manifest_path}. '
                'Every skill project must include a skill.yaml file.'
            )
            raise FileNotFoundError(msg)
        return SkillManifest.model_validate(_parse_yaml(manifest_path))

    def _read_channel_manifest(self, repo_dir: Path) -> ChannelManifest:
        """
        title: Read and validate a channel manifest and skill layout.
        parameters:
          repo_dir:
            type: Path
        returns:
          type: ChannelManifest
        """
        manifest_path = repo_dir / 'skills-channel.yaml'
        if not manifest_path.exists():
            msg = f'No skills-channel.yaml found in {repo_dir}.'
            raise FileNotFoundError(msg)

        manifest = ChannelManifest.model_validate(_parse_yaml(manifest_path))
        if not manifest.skills:
            msg = (
                'skills-channel.yaml must declare at least one installable '
                'skill.'
            )
            raise ValueError(msg)

        for declared in manifest.skills:
            skill_dir = self._skill_dir(repo_dir, declared.name)
            skill_manifest = self._skill_manifest_path(repo_dir, declared.name)
            if not skill_dir.is_dir():
                relative_dir = skill_dir.relative_to(repo_dir)
                msg = (
                    f'Declared skill {declared.name!r} is missing its '
                    f'expected directory {relative_dir!s} in {repo_dir}.'
                )
                raise ValueError(msg)
            if not skill_manifest.is_file():
                relative_manifest = skill_manifest.relative_to(repo_dir)
                msg = (
                    f'Declared skill {declared.name!r} is missing its '
                    f'expected manifest {relative_manifest!s} in {repo_dir}.'
                )
                raise ValueError(msg)
            self._read_skill_manifest_file(skill_manifest)

        return manifest

    def _detect_source_kind(self, repo_dir: Path) -> str:
        """
        title: Detect the registry source type for a materialized repo.
        parameters:
          repo_dir:
            type: Path
        returns:
          type: str
        """
        if (repo_dir / 'skills-channel.yaml').exists():
            return 'channel'
        msg = (
            f'Cannot register {repo_dir}. Expected skills-channel.yaml at the '
            'channel root.'
        )
        raise ValueError(msg)

    def _validate_local_name(
        self, local_name: str, state: RegistryState
    ) -> str:
        """
        title: Validate a local channel alias for registration.
        parameters:
          local_name:
            type: str
          state:
            type: RegistryState
        returns:
          type: str
        """
        value = local_name.strip()
        if not value:
            raise ValueError('local_name must not be empty.')
        if '.' in value or not _LOCAL_NAME_PATTERN.fullmatch(value):
            msg = (
                'local_name must contain only letters, numbers, "_" or "-", '
                'and it must not contain ".".'
            )
            raise ValueError(msg)
        if value == BUILTIN_CHANNEL:
            msg = (
                f'local_name {value!r} is reserved for built-in hiperhealth '
                'skills.'
            )
            raise ValueError(msg)
        if value in state.channels:
            msg = f'Channel alias {value!r} is already registered.'
            raise ValueError(msg)
        return value

    def _resolve_local_name(
        self,
        channel_manifest: ChannelManifest,
        local_name: str | None,
        state: RegistryState,
    ) -> str:
        """
        title: Resolve the local alias used to store a channel.
        parameters:
          channel_manifest:
            type: ChannelManifest
          local_name:
            type: str | None
          state:
            type: RegistryState
        returns:
          type: str
        """
        candidate = (
            local_name
            or channel_manifest.channel.default_alias
            or channel_manifest.channel.name
        )
        if candidate is None:
            msg = (
                'local_name is required because the channel did not declare '
                'a default_alias or name.'
            )
            raise ValueError(msg)
        return self._validate_local_name(candidate, state)

    def _detect_provider(self, source: str) -> str:
        """
        title: Infer the provider label for a channel source string.
        parameters:
          source:
            type: str
        returns:
          type: str
        """
        if Path(source).exists():
            return 'local'
        parsed = urlparse(source)
        host = parsed.netloc.lower()
        if not host and source.startswith('git@'):
            host = source.split('@', maxsplit=1)[1].split(':', maxsplit=1)[0]
        if 'github' in host:
            return 'github'
        if 'gitlab' in host:
            return 'gitlab'
        if host:
            return 'git'
        return 'local'

    def _looks_like_git_source(self, source: str) -> bool:
        """
        title: Check whether a source string looks like a Git URL.
        parameters:
          source:
            type: str
        returns:
          type: bool
        """
        return source.startswith(
            ('https://', 'http://', 'git@', 'ssh://', 'file://')
        )

    def _copy_source_tree(self, source_dir: Path, target_dir: Path) -> None:
        """
        title: Copy a local folder source into the channel checkout path.
        parameters:
          source_dir:
            type: Path
          target_dir:
            type: Path
        """
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

    def _materialize_channel_source(
        self, source: str, target_dir: Path, ref: str | None = None
    ) -> tuple[str, bool]:
        """
        title: Copy or clone a channel source into a working directory.
        parameters:
          source:
            type: str
          target_dir:
            type: Path
          ref:
            type: str | None
        returns:
          type: tuple[str, bool]
        """
        source_path = Path(source).expanduser()
        if source_path.exists():
            if not source_path.is_dir():
                msg = (
                    f'Cannot register channel from {source!r}. Provide a '
                    'directory path or a Git URL.'
                )
                raise ValueError(msg)
            if ref is not None:
                msg = 'ref is only supported for remote git sources.'
                raise ValueError(msg)
            resolved_source = source_path.resolve()
            self._copy_source_tree(resolved_source, target_dir)
            return str(resolved_source), True

        if not self._looks_like_git_source(source):
            msg = (
                f'Cannot register channel from {source!r}. Provide a local '
                'directory path or a Git URL.'
            )
            raise ValueError(msg)

        self._clone_repo(source, target_dir, ref=ref)
        return source, False

    def _clone_repo(
        self, source: str, target_dir: Path, ref: str | None = None
    ) -> None:
        """
        title: Clone a remote Git repository for a channel.
        parameters:
          source:
            type: str
          target_dir:
            type: Path
          ref:
            type: str | None
        """
        self._run_command(['git', 'clone', source, str(target_dir)])
        if ref is not None:
            self._run_command(
                ['git', 'checkout', ref],
                cwd=target_dir,
            )

    def _current_commit(self, repo_dir: Path) -> str:
        """
        title: Return the current Git commit for a channel checkout.
        parameters:
          repo_dir:
            type: Path
        returns:
          type: str
        """
        try:
            return self._run_command(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_dir,
            )
        except subprocess.CalledProcessError:
            return ''

    def _current_ref(self, repo_dir: Path) -> str | None:
        """
        title: Return the current Git branch name when available.
        parameters:
          repo_dir:
            type: Path
        returns:
          type: str | None
        """
        try:
            ref = self._run_command(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=repo_dir,
            )
        except subprocess.CalledProcessError:
            return None
        return None if ref == 'HEAD' else ref

    def _update_repo(self, repo_dir: Path, ref: str | None = None) -> str:
        """
        title: Fetch and fast-forward a remote channel checkout.
        parameters:
          repo_dir:
            type: Path
          ref:
            type: str | None
        returns:
          type: str
        """
        self._run_command(
            ['git', 'fetch', 'origin', '--tags', '--prune'],
            cwd=repo_dir,
        )
        if ref is not None:
            self._run_command(['git', 'checkout', ref], cwd=repo_dir)
            try:
                self._run_command(
                    ['git', 'pull', '--ff-only', 'origin', ref],
                    cwd=repo_dir,
                )
            except subprocess.CalledProcessError:
                pass
        else:
            try:
                self._run_command(['git', 'pull', '--ff-only'], cwd=repo_dir)
            except subprocess.CalledProcessError:
                pass
        return self._current_commit(repo_dir)

    def _channel_record_from_repo(
        self,
        local_name: str,
        source: str,
        repo_dir: Path,
        registered_at: str | None = None,
        ref: str | None = None,
    ) -> ChannelRecord:
        """
        title: Build persisted channel metadata from a checkout.
        parameters:
          local_name:
            type: str
          source:
            type: str
          repo_dir:
            type: Path
          registered_at:
            type: str | None
          ref:
            type: str | None
        returns:
          type: ChannelRecord
        """
        channel_manifest = self._read_channel_manifest(repo_dir)
        ref_value = ref if ref is not None else self._current_ref(repo_dir)
        timestamp = _utcnow()
        available = [
            _canonical_skill_id(local_name, skill.name)
            for skill in channel_manifest.skills
        ]
        return ChannelRecord(
            local_name=local_name,
            remote_name=channel_manifest.channel.name,
            provider=self._detect_provider(source),
            source=source,
            ref=ref_value,
            commit=self._current_commit(repo_dir),
            registered_at=registered_at or timestamp,
            updated_at=timestamp,
            available_skills=sorted(available),
        )

    def _iter_builtin_skill_entries(
        self,
    ) -> Iterator[tuple[Path, SkillManifest]]:
        """
        title: Iterate over built-in skill directories and manifests.
        returns:
          type: Iterator[tuple[Path, SkillManifest]]
        """
        if not self._builtin_dir.is_dir():
            return
        for child in sorted(self._builtin_dir.iterdir()):
            manifest_path = child / 'skill.yaml'
            if manifest_path.exists():
                yield child, self._read_skill_manifest_file(manifest_path)

    def _builtin_skill_name(
        self, manifest: SkillManifest, skill_dir: Path
    ) -> str:
        """
        title: Normalize the visible name for a built-in skill.
        parameters:
          manifest:
            type: SkillManifest
          skill_dir:
            type: Path
        returns:
          type: str
        """
        prefix = f'{BUILTIN_CHANNEL}.'
        if manifest.name.startswith(prefix):
            return manifest.name[len(prefix) :]
        return skill_dir.name

    def _iter_channel_skill_entries(
        self, local_name: str
    ) -> list[_ResolvedChannelSkill]:
        """
        title: Resolve all declared skills for a registered channel.
        parameters:
          local_name:
            type: str
        returns:
          type: list[_ResolvedChannelSkill]
        """
        repo_dir = self._channel_repo_dir(local_name)
        if not repo_dir.is_dir():
            msg = f'Channel {local_name!r} is not registered.'
            raise KeyError(msg)

        channel_manifest = self._read_channel_manifest(repo_dir)
        resolved: list[_ResolvedChannelSkill] = []
        for declared in channel_manifest.skills:
            skill_dir = self._skill_dir(repo_dir, declared.name)
            manifest_path = self._skill_manifest_path(repo_dir, declared.name)
            manifest = self._read_skill_manifest_file(manifest_path)
            resolved.append(
                _ResolvedChannelSkill(
                    available=AvailableSkillRecord(
                        channel=local_name,
                        name=declared.name,
                        canonical_id=_canonical_skill_id(
                            local_name, declared.name
                        ),
                        path=str(skill_dir.relative_to(repo_dir)),
                        manifest_path=str(manifest_path.relative_to(repo_dir)),
                        enabled=declared.enabled,
                        tags=list(declared.tags),
                    ),
                    manifest=manifest,
                    manifest_path=manifest_path,
                )
            )
        return resolved

    def _find_available_channel_skill(
        self, skill_id: str
    ) -> _ResolvedChannelSkill | None:
        """
        title: Resolve an available channel skill by canonical id.
        parameters:
          skill_id:
            type: str
        returns:
          type: _ResolvedChannelSkill | None
        """
        local_name, separator, skill_name = skill_id.partition('.')
        if not separator:
            return None

        state = self._load_state()
        if local_name not in state.channels:
            return None
        for entry in self._iter_channel_skill_entries(local_name):
            if entry.available.name == skill_name:
                return entry
        return None

    def _install_dependencies(self, dependencies: list[str]) -> None:
        """
        title: Install Python dependencies declared by a skill.
        parameters:
          dependencies:
            type: list[str]
        """
        if not dependencies:
            return
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', *dependencies],
            check=True,
            capture_output=True,
            text=True,
        )

    def _normalize_loaded_skill(
        self, skill: BaseSkill, canonical_id: str
    ) -> BaseSkill:
        """
        title: Rewrite loaded skill metadata to use its canonical id.
        parameters:
          skill:
            type: BaseSkill
          canonical_id:
            type: str
        returns:
          type: BaseSkill
        """
        skill.metadata = SkillMetadata(
            name=canonical_id,
            version=skill.metadata.version,
            stages=tuple(skill.metadata.stages),
            description=skill.metadata.description,
        )
        return skill

    def _load_channel_skill(self, record: InstalledSkillRecord) -> BaseSkill:
        """
        title: Load an installed channel skill from its manifest.
        parameters:
          record:
            type: InstalledSkillRecord
        returns:
          type: BaseSkill
        """
        manifest_path = Path(record.manifest_path)
        manifest = self._read_skill_manifest_file(manifest_path)
        skill_dir = manifest_path.parent
        cls = _load_class_from_directory(skill_dir, manifest.entry_point)
        skill = cls()
        return self._normalize_loaded_skill(skill, record.id)

    def add_channel(
        self,
        source: str,
        local_name: str | None = None,
        ref: str | None = None,
    ) -> str:
        """
        title: Register a new local-folder or Git-backed skill channel.
        parameters:
          source:
            type: str
          local_name:
            type: str | None
          ref:
            type: str | None
        returns:
          type: str
        """
        self._ensure_storage_dirs()
        state = self._load_state()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_repo = Path(tmp_dir) / 'repo'
            normalized_source, is_local = self._materialize_channel_source(
                source,
                temp_repo,
                ref=ref,
            )
            self._detect_source_kind(temp_repo)
            channel_manifest = self._read_channel_manifest(temp_repo)
            resolved_name = self._resolve_local_name(
                channel_manifest, local_name, state
            )
            target_repo = self._channel_repo_dir(resolved_name)
            if target_repo.exists():
                msg = f'Channel alias {resolved_name!r} is already registered.'
                raise ValueError(msg)
            target_repo.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_repo), str(target_repo))

        record = self._channel_record_from_repo(
            resolved_name,
            normalized_source,
            target_repo,
            ref=None if is_local else ref,
        )
        state.channels[resolved_name] = record
        self._save_state(state)
        self._save_channel_metadata(resolved_name, record, target_repo)
        return resolved_name

    def list_channels(self) -> list[ChannelRecord]:
        """
        title: List all registered channels.
        returns:
          type: list[ChannelRecord]
        """
        state = self._load_state()
        return [state.channels[name] for name in sorted(state.channels.keys())]

    def list_channel_skills(
        self, local_name: str
    ) -> list[AvailableSkillRecord]:
        """
        title: List the skills declared by one registered channel.
        parameters:
          local_name:
            type: str
        returns:
          type: list[AvailableSkillRecord]
        """
        state = self._load_state()
        if local_name not in state.channels:
            msg = f'Channel {local_name!r} is not registered.'
            raise KeyError(msg)
        entries = self._iter_channel_skill_entries(local_name)
        return [
            entry.available
            for entry in sorted(
                entries,
                key=lambda entry: entry.available.canonical_id,
            )
        ]

    def update_channel(
        self, local_name: str, ref: str | None = None
    ) -> list[str]:
        """
        title: Refresh a channel checkout and its installed skills.
        parameters:
          local_name:
            type: str
          ref:
            type: str | None
        returns:
          type: list[str]
        """
        state = self._load_state()
        channel = state.channels.get(local_name)
        if channel is None:
            msg = f'Channel {local_name!r} is not registered.'
            raise KeyError(msg)

        repo_dir = self._channel_repo_dir(local_name)
        target_ref: str | None = None
        if channel.provider == 'local':
            if ref is not None:
                msg = 'ref is only supported for remote git sources.'
                raise ValueError(msg)
            source_dir = Path(channel.source)
            if not source_dir.is_dir():
                msg = (
                    f'Local channel source {channel.source!r} no longer '
                    'exists.'
                )
                raise FileNotFoundError(msg)
            self._copy_source_tree(source_dir, repo_dir)
        else:
            target_ref = ref if ref is not None else channel.ref
            self._update_repo(repo_dir, ref=target_ref)
        refreshed_channel = self._channel_record_from_repo(
            local_name,
            channel.source,
            repo_dir,
            registered_at=channel.registered_at,
            ref=target_ref,
        )
        state.channels[local_name] = refreshed_channel

        available_map = {
            entry.available.canonical_id: entry
            for entry in self._iter_channel_skill_entries(local_name)
        }
        updated: list[str] = []
        for skill_id, record in list(state.skills.items()):
            if record.channel != local_name:
                continue
            available = available_map.get(skill_id)
            if available is None:
                state.skills.pop(skill_id, None)
                continue

            self._install_dependencies(available.manifest.dependencies)
            state.skills[skill_id] = InstalledSkillRecord(
                id=skill_id,
                channel=local_name,
                skill_name=available.available.name,
                manifest_path=str(available.manifest_path),
                installed_at=record.installed_at,
                updated_at=_utcnow(),
                version=available.manifest.version,
                source_commit=refreshed_channel.commit,
                enabled=available.available.enabled,
            )
            updated.append(skill_id)

        self._save_state(state)
        self._save_channel_metadata(local_name, refreshed_channel, repo_dir)
        return sorted(updated)

    def remove_channel(self, local_name: str) -> None:
        """
        title: Remove a registered channel and its installed skills.
        parameters:
          local_name:
            type: str
        """
        state = self._load_state()
        if local_name not in state.channels:
            msg = f'Channel {local_name!r} is not registered.'
            raise KeyError(msg)

        for skill_id, record in list(state.skills.items()):
            if record.channel != local_name:
                continue
            state.skills.pop(skill_id, None)

        shutil.rmtree(self._channel_dir(local_name), ignore_errors=True)
        state.channels.pop(local_name, None)
        self._save_state(state)

    def list_skills(
        self,
        channel: str | None = None,
        installed_only: bool = False,
    ) -> list[SkillSummary]:
        """
        title: List built-in and channel skills known to the registry.
        parameters:
          channel:
            type: str | None
          installed_only:
            type: bool
        returns:
          type: list[SkillSummary]
        """
        state = self._load_state()
        summaries: list[SkillSummary] = []

        if channel in (None, BUILTIN_CHANNEL):
            for skill_dir, manifest in self._iter_builtin_skill_entries():
                skill_name = self._builtin_skill_name(manifest, skill_dir)
                canonical_id = _canonical_skill_id(BUILTIN_CHANNEL, skill_name)
                summaries.append(
                    SkillSummary(
                        **manifest.model_dump(),
                        channel=BUILTIN_CHANNEL,
                        skill_name=skill_name,
                        canonical_id=canonical_id,
                        manifest_path=str(skill_dir / 'skill.yaml'),
                        installed=True,
                        enabled=True,
                        builtin=True,
                    )
                )

        if channel is None:
            channel_names = sorted(state.channels.keys())
        elif channel in state.channels:
            channel_names = [channel]
        else:
            channel_names = []

        for local_name in channel_names:
            installed_ids = {
                skill_id
                for skill_id, record in state.skills.items()
                if record.channel == local_name
            }
            for entry in self._iter_channel_skill_entries(local_name):
                if (
                    installed_only
                    and entry.available.canonical_id not in installed_ids
                ):
                    continue
                summaries.append(
                    SkillSummary(
                        **entry.manifest.model_dump(),
                        channel=local_name,
                        skill_name=entry.available.name,
                        canonical_id=entry.available.canonical_id,
                        manifest_path=str(entry.manifest_path),
                        installed=entry.available.canonical_id
                        in installed_ids,
                        enabled=entry.available.enabled,
                        tags=list(entry.available.tags),
                    )
                )

        unique: dict[str, SkillSummary] = {
            summary.canonical_id: summary for summary in summaries
        }
        return [unique[key] for key in sorted(unique.keys())]

    def install_skill(self, skill_id: str) -> str:
        """
        title: Install one skill from a registered channel.
        parameters:
          skill_id:
            type: str
        returns:
          type: str
        """
        if skill_id.startswith(f'{BUILTIN_CHANNEL}.'):
            msg = 'Built-in hiperhealth skills do not need installation.'
            raise ValueError(msg)

        state = self._load_state()
        available = self._find_available_channel_skill(skill_id)
        if available is None:
            msg = (
                f'Skill {skill_id!r} is not available from any registered '
                'channel.'
            )
            raise KeyError(msg)

        existing = state.skills.get(skill_id)
        channel = state.channels[available.available.channel]
        installed_at = existing.installed_at if existing else _utcnow()
        state.skills[skill_id] = InstalledSkillRecord(
            id=skill_id,
            channel=available.available.channel,
            skill_name=available.available.name,
            manifest_path=str(available.manifest_path),
            installed_at=installed_at,
            updated_at=_utcnow(),
            version=available.manifest.version,
            source_commit=channel.commit,
            enabled=available.available.enabled,
        )
        self._save_state(state)
        self._install_dependencies(available.manifest.dependencies)
        return skill_id

    def install_channel(
        self, local_name: str, include_disabled: bool = False
    ) -> list[str]:
        """
        title: Install all eligible skills from a registered channel.
        parameters:
          local_name:
            type: str
          include_disabled:
            type: bool
        returns:
          type: list[str]
        """
        state = self._load_state()
        if local_name not in state.channels:
            msg = f'Channel {local_name!r} is not registered.'
            raise KeyError(msg)

        installed: list[str] = []
        for entry in self._iter_channel_skill_entries(local_name):
            if not include_disabled and not entry.available.enabled:
                continue
            installed.append(self.install_skill(entry.available.canonical_id))
        return sorted(installed)

    def update_skill(self, skill_id: str, pull_channel: bool = False) -> str:
        """
        title: Refresh one installed skill, optionally pulling its channel.
        parameters:
          skill_id:
            type: str
          pull_channel:
            type: bool
        returns:
          type: str
        """
        state = self._load_state()
        record = state.skills.get(skill_id)
        if record is None:
            msg = f'Skill {skill_id!r} is not installed.'
            raise KeyError(msg)

        if record.channel is None:
            msg = f'Skill {skill_id!r} has no owning channel.'
            raise ValueError(msg)

        if pull_channel:
            self.update_channel(record.channel)
            return skill_id

        available = self._find_available_channel_skill(skill_id)
        if available is None:
            msg = (
                f'Skill {skill_id!r} is no longer declared by channel '
                f'{record.channel!r}.'
            )
            raise KeyError(msg)

        channel = state.channels[record.channel]
        state.skills[skill_id] = InstalledSkillRecord(
            id=skill_id,
            channel=record.channel,
            skill_name=available.available.name,
            manifest_path=str(available.manifest_path),
            installed_at=record.installed_at,
            updated_at=_utcnow(),
            version=available.manifest.version,
            source_commit=channel.commit,
            enabled=available.available.enabled,
        )
        self._save_state(state)
        self._install_dependencies(available.manifest.dependencies)
        return skill_id

    def remove_skill(self, skill_id: str) -> None:
        """
        title: Remove one installed channel skill from the registry.
        parameters:
          skill_id:
            type: str
        """
        if skill_id.startswith(f'{BUILTIN_CHANNEL}.'):
            msg = 'Built-in hiperhealth skills cannot be removed.'
            raise ValueError(msg)

        state = self._load_state()
        record = state.skills.get(skill_id)
        if record is None:
            msg = f'Skill {skill_id!r} is not installed.'
            raise KeyError(msg)

        state.skills.pop(skill_id, None)
        self._save_state(state)

    def load(self, name: str) -> BaseSkill:
        """
        title: Load a built-in or installed channel skill by name.
        parameters:
          name:
            type: str
        returns:
          type: BaseSkill
        """
        for skill_dir, manifest in self._iter_builtin_skill_entries():
            if manifest.name != name:
                continue
            package_base = f'hiperhealth.skills.{skill_dir.name}'
            cls = _load_class_from_package(package_base, manifest.entry_point)
            skill = cls()
            return self._normalize_loaded_skill(skill, manifest.name)

        state = self._load_state()
        record = state.skills.get(name)
        if record is not None:
            return self._load_channel_skill(record)

        if self._find_available_channel_skill(name) is not None:
            msg = (
                f'Skill {name!r} is available but not installed. Use '
                f'install_skill({name!r}) first.'
            )
            raise KeyError(msg)

        msg = (
            f'Skill {name!r} not found. Use list_skills() to inspect '
            'available skills and install_skill() or install_channel() to add '
            'them.'
        )
        raise KeyError(msg)

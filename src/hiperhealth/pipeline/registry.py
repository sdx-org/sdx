"""
title: Channel-aware skill registry for built-in, channel, and legacy skills.
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
    """
    Metadata parsed from a skill ``hiperhealth.yaml`` file.

    Parameters
    ----------
    api_version:
        Manifest schema version.
    name:
        Skill name declared by the manifest.
    version:
        Skill version string.
    entry_point:
        Import target in ``module:ClassName`` format.
    stages:
        Pipeline stages handled by the skill.
    description:
        Human-readable description.
    author:
        Author information.
    license:
        License identifier.
    homepage:
        Project homepage.
    min_hiperhealth_version:
        Minimum supported hiperhealth version string.
    dependencies:
        Extra Python package dependencies.
    """

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
    """
    Root metadata stored in ``skills.yaml``.

    Parameters
    ----------
    name:
        Canonical remote channel name.
    display_name:
        Human-readable channel name.
    default_alias:
        Suggested local alias for the channel.
    version:
        Channel metadata version.
    description:
        Channel description.
    homepage:
        Channel homepage.
    license:
        Channel license identifier.
    min_hiperhealth_version:
        Minimum supported hiperhealth version string.
    """

    name: str
    display_name: str = ''
    default_alias: str | None = None
    version: str = ''
    description: str = ''
    homepage: str = ''
    license: str = ''
    min_hiperhealth_version: str = ''


class ChannelDiscovery(BaseModel):
    """
    Discovery hints stored in ``skills.yaml``.

    Parameters
    ----------
    skills_dir:
        Directory containing installable skills.
    ignore:
        Repository paths that are not installable skills.
    """

    skills_dir: str = 'skills'
    ignore: list[str] = Field(default_factory=list)


class DeclaredSkill(BaseModel):
    """
    One installable skill declaration inside ``skills.yaml``.

    Parameters
    ----------
    name:
        Channel-local skill name.
    path:
        Relative path to the skill directory.
    manifest:
        Relative path to the skill manifest.
    enabled:
        Whether the skill participates in default channel installs.
    tags:
        Optional skill tags.
    """

    name: str
    path: str
    manifest: str
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)


class ChannelManifest(BaseModel):
    """
    Parsed root-level ``skills.yaml`` manifest.

    Parameters
    ----------
    api_version:
        Manifest schema version.
    channel:
        Channel metadata block.
    discovery:
        Discovery configuration block.
    skills:
        Declared installable skills.
    """

    api_version: int = 1
    channel: ChannelMetadata
    discovery: ChannelDiscovery = Field(default_factory=ChannelDiscovery)
    skills: list[DeclaredSkill] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_skill_names(self) -> ChannelManifest:
        names = [skill.name for skill in self.skills]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            joined = ', '.join(duplicates)
            msg = f'Duplicate skill names declared in skills.yaml: {joined}.'
            raise ValueError(msg)
        return self


class ChannelRecord(BaseModel):
    """
    Persisted metadata for one registered channel.

    Parameters
    ----------
    local_name:
        User-defined local alias.
    remote_name:
        Channel name from ``skills.yaml``.
    provider:
        Source provider name such as ``github`` or ``gitlab``.
    source:
        Original source URL or local git path.
    ref:
        Tracking git reference, if any.
    commit:
        Current checked-out commit hash.
    registered_at:
        Initial registration timestamp in UTC.
    updated_at:
        Last update timestamp in UTC.
    available_skills:
        Available canonical skill identifiers.
    """

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
    """
    One installable skill exposed by a channel.

    Parameters
    ----------
    channel:
        Local channel alias.
    name:
        Channel-local skill name.
    canonical_id:
        Canonical skill identifier.
    path:
        Relative path to the skill directory.
    manifest_path:
        Relative path to the skill manifest.
    enabled:
        Whether the skill participates in default channel installs.
    tags:
        Optional skill tags.
    """

    channel: str
    name: str
    canonical_id: str
    path: str
    manifest_path: str
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)


class InstalledSkillRecord(BaseModel):
    """
    Persisted metadata for one installed skill.

    Parameters
    ----------
    id:
        Canonical installed skill identifier.
    channel:
        Owning channel alias, if any.
    skill_name:
        Channel-local skill name or legacy skill name.
    manifest_path:
        Absolute path to the active skill manifest.
    installed_at:
        Installation timestamp in UTC.
    updated_at:
        Last refresh timestamp in UTC.
    version:
        Installed skill version.
    source_commit:
        Source commit hash for channel installs.
    enabled:
        Whether the source declaration is enabled.
    source:
        Original source URL or local path.
    artifact_path:
        Optional copied artifact path for legacy installs.
    legacy:
        Whether this record comes from a legacy single-skill install.
    """

    id: str
    channel: str | None = None
    skill_name: str
    manifest_path: str
    installed_at: str
    updated_at: str
    version: str
    source_commit: str = ''
    enabled: bool = True
    source: str = ''
    artifact_path: str | None = None
    legacy: bool = False


class RegistryState(BaseModel):
    """
    Persisted registry state.

    Parameters
    ----------
    channels:
        Registered channels keyed by local alias.
    skills:
        Installed skills keyed by canonical identifier.
    """

    channels: dict[str, ChannelRecord] = Field(default_factory=dict)
    skills: dict[str, InstalledSkillRecord] = Field(default_factory=dict)


class SkillSummary(SkillManifest):
    """
    Skill manifest plus registry-specific summary data.

    Parameters
    ----------
    channel:
        Owning channel alias. ``None`` for legacy installs.
    skill_name:
        Channel-local skill name or legacy skill name.
    canonical_id:
        Canonical identifier used by registry operations.
    manifest_path:
        Absolute path to the manifest file.
    installed:
        Whether the skill is installed locally.
    enabled:
        Whether the skill is enabled in its source declaration.
    builtin:
        Whether the skill ships with hiperhealth itself.
    tags:
        Optional channel-level skill tags.
    """

    channel: str | None = None
    skill_name: str
    canonical_id: str
    manifest_path: str
    installed: bool = False
    enabled: bool = True
    builtin: bool = False
    tags: list[str] = Field(default_factory=list)


class _LegacyInstalledSkillRecord(BaseModel):
    source: str
    installed_at: str
    version: str


class _LegacyRegistryIndex(BaseModel):
    skills: dict[str, _LegacyInstalledSkillRecord] = Field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _ResolvedChannelSkill:
    available: AvailableSkillRecord
    manifest: SkillManifest
    manifest_path: Path


def _parse_yaml(path: Path) -> dict[str, Any]:
    """
    Parse a YAML file into a dictionary.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.
    """
    import yaml

    with path.open(encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def _utcnow() -> str:
    """
    Return the current UTC timestamp in ISO 8601 format.

    Returns
    -------
    str
        Current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _canonical_skill_id(local_name: str, skill_name: str) -> str:
    """
    Build a canonical skill identifier.

    Parameters
    ----------
    local_name:
        Local channel alias.
    skill_name:
        Channel-local skill name.

    Returns
    -------
    str
        Canonical skill identifier.
    """
    return f'{local_name}.{skill_name}'


def _split_entry_point(entry_point: str) -> tuple[str, str]:
    """
    Split an entry point into module and class names.

    Parameters
    ----------
    entry_point:
        Entry point in ``module:ClassName`` format.

    Returns
    -------
    tuple[str, str]
        Module name and class name.
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
    Temporarily prepend a directory to ``sys.path``.

    Parameters
    ----------
    path:
        Directory to prepend.

    Yields
    ------
    None
        Context manager sentinel.
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
    Dynamically import a skill class from a skill directory.

    Parameters
    ----------
    skill_dir:
        Root skill directory.
    entry_point:
        Entry point in ``module:ClassName`` format.

    Returns
    -------
    type[Any]
        Loaded class object.
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
    Dynamically import a built-in skill class from the package.

    Parameters
    ----------
    package_base:
        Built-in package prefix.
    entry_point:
        Entry point in ``module:ClassName`` format.

    Returns
    -------
    type[Any]
        Loaded class object.
    """
    module_name, class_name = _split_entry_point(entry_point)
    module = importlib.import_module(f'{package_base}.{module_name}')
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        msg = f'Entry point {entry_point!r} does not resolve to a class.'
        raise TypeError(msg)
    return cls


class SkillRegistry:
    """
    Manage built-in, channel-based, and legacy single-skill registries.

    Parameters
    ----------
    registry_dir:
        Optional artifact directory for installed skills. When omitted,
        ``~/.hiperhealth/artifacts/skills`` is used.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
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
        Return the legacy-compatible skill artifact directory.

        Returns
        -------
        Path
            Installed skill artifact directory.
        """
        return self._registry_dir

    @property
    def root_dir(self) -> Path:
        """
        Return the registry root directory.

        Returns
        -------
        Path
            Registry root directory.
        """
        return self._root_dir

    def _ensure_storage_dirs(self) -> None:
        """
        Create the registry storage directories when missing.
        """
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._channels_dir.mkdir(parents=True, exist_ok=True)
        self._registry_dir.mkdir(parents=True, exist_ok=True)

    def _run_command(self, args: list[str], cwd: Path | None = None) -> str:
        """
        Run a subprocess command and return stripped stdout.

        Parameters
        ----------
        args:
            Command arguments.
        cwd:
            Optional working directory.

        Returns
        -------
        str
            Command stdout.
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
        return self._state_dir / 'state.json'

    def _channels_index_path(self) -> Path:
        return self._state_dir / 'channels.json'

    def _skills_index_path(self) -> Path:
        return self._state_dir / 'skills.json'

    def _channel_dir(self, local_name: str) -> Path:
        return self._channels_dir / local_name

    def _channel_repo_dir(self, local_name: str) -> Path:
        return self._channel_dir(local_name) / 'repo'

    def _channel_manifest_copy_path(self, local_name: str) -> Path:
        return self._channel_dir(local_name) / 'skills.yaml'

    def _channel_record_path(self, local_name: str) -> Path:
        return self._channel_dir(local_name) / 'channel.json'

    def _artifact_dir(self, skill_id: str) -> Path:
        return self._registry_dir / skill_id

    def _load_state(self) -> RegistryState:
        """
        Load registry state from disk.

        Returns
        -------
        RegistryState
            Current registry state.
        """
        self._maybe_migrate_legacy_registry()
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
        Persist registry state and split indexes to disk.

        Parameters
        ----------
        state:
            Registry state to persist.
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
        Persist metadata files for one registered channel.

        Parameters
        ----------
        local_name:
            Local channel alias.
        record:
            Channel record to persist.
        repo_dir:
            Channel repository checkout.
        """
        channel_dir = self._channel_dir(local_name)
        channel_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            repo_dir / 'skills.yaml',
            self._channel_manifest_copy_path(local_name),
        )
        self._channel_record_path(local_name).write_text(
            record.model_dump_json(indent=2) + '\n',
            encoding='utf-8',
        )

    def _maybe_migrate_legacy_registry(self) -> None:
        """
        Import legacy flat registry metadata when present.
        """
        state_path = self._state_path()
        if state_path.exists():
            return

        manifest_path = self._registry_dir / 'manifest.json'
        if not manifest_path.exists():
            return

        legacy_index = _LegacyRegistryIndex.model_validate_json(
            manifest_path.read_text(encoding='utf-8')
        )
        state = RegistryState()
        for skill_id, legacy_record in legacy_index.skills.items():
            manifest_path_value = (
                self._artifact_dir(skill_id) / 'hiperhealth.yaml'
            )
            if not manifest_path_value.exists():
                continue
            state.skills[skill_id] = InstalledSkillRecord(
                id=skill_id,
                skill_name=skill_id,
                manifest_path=str(manifest_path_value),
                installed_at=legacy_record.installed_at,
                updated_at=legacy_record.installed_at,
                version=legacy_record.version,
                source=legacy_record.source,
                artifact_path=str(self._artifact_dir(skill_id)),
                legacy=True,
            )
        self._save_state(state)

    def _read_skill_manifest_file(self, manifest_path: Path) -> SkillManifest:
        """
        Read and validate a skill manifest file.

        Parameters
        ----------
        manifest_path:
            Path to ``hiperhealth.yaml``.

        Returns
        -------
        SkillManifest
            Validated skill manifest.
        """
        if not manifest_path.exists():
            msg = (
                f'No hiperhealth.yaml found at {manifest_path}. '
                'Every skill project must include a hiperhealth.yaml file.'
            )
            raise FileNotFoundError(msg)
        return SkillManifest.model_validate(_parse_yaml(manifest_path))

    def _read_manifest(self, skill_dir: Path) -> SkillManifest:
        """
        Read and validate a skill manifest from a directory.

        Parameters
        ----------
        skill_dir:
            Skill directory containing ``hiperhealth.yaml``.

        Returns
        -------
        SkillManifest
            Validated skill manifest.
        """
        return self._read_skill_manifest_file(skill_dir / 'hiperhealth.yaml')

    def _read_channel_manifest(self, repo_dir: Path) -> ChannelManifest:
        """
        Read and validate a channel manifest.

        Parameters
        ----------
        repo_dir:
            Repository root directory.

        Returns
        -------
        ChannelManifest
            Validated channel manifest.
        """
        manifest_path = repo_dir / 'skills.yaml'
        if not manifest_path.exists():
            msg = f'No skills.yaml found in {repo_dir}.'
            raise FileNotFoundError(msg)

        manifest = ChannelManifest.model_validate(_parse_yaml(manifest_path))
        if not manifest.skills:
            msg = 'skills.yaml must declare at least one installable skill.'
            raise ValueError(msg)

        for declared in manifest.skills:
            skill_dir = repo_dir / declared.path
            skill_manifest = repo_dir / declared.manifest
            if not skill_dir.is_dir():
                msg = (
                    f'Declared skill path {declared.path!r} does not exist in '
                    f'{repo_dir}.'
                )
                raise ValueError(msg)
            if not skill_manifest.is_file():
                msg = (
                    f'Declared manifest {declared.manifest!r} '
                    'does not exist in '
                    f'{repo_dir}.'
                )
                raise ValueError(msg)
            self._read_skill_manifest_file(skill_manifest)

        return manifest

    def _detect_source_kind(self, repo_dir: Path) -> str:
        """
        Detect whether a repository is a channel or legacy skill source.

        Parameters
        ----------
        repo_dir:
            Repository root directory.

        Returns
        -------
        str
            ``"channel"`` or ``"legacy"``.
        """
        if (repo_dir / 'skills.yaml').exists():
            return 'channel'
        if (repo_dir / 'hiperhealth.yaml').exists():
            return 'legacy'
        msg = (
            f'Cannot register {repo_dir}. Expected skills.yaml for a channel '
            'repository or hiperhealth.yaml for a legacy single-skill '
            'repository.'
        )
        raise ValueError(msg)

    def _validate_local_name(
        self, local_name: str, state: RegistryState
    ) -> str:
        """
        Validate and normalize a local channel alias.

        Parameters
        ----------
        local_name:
            Proposed local channel alias.
        state:
            Current registry state.

        Returns
        -------
        str
            Validated local alias.
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
        Resolve the final local channel alias.

        Parameters
        ----------
        channel_manifest:
            Parsed channel manifest.
        local_name:
            User-provided local alias, if any.
        state:
            Current registry state.

        Returns
        -------
        str
            Final local alias.
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
        Detect the provider type for a git source string.

        Parameters
        ----------
        source:
            Source URL or local path.

        Returns
        -------
        str
            Provider name.
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
        Return whether a source looks cloneable by git.

        Parameters
        ----------
        source:
            Source URL or local path.

        Returns
        -------
        bool
            Whether the source is git-like.
        """
        path = Path(source)
        return path.exists() or source.startswith(
            ('https://', 'http://', 'git@', 'ssh://', 'file://')
        )

    def _clone_repo(
        self, source: str, target_dir: Path, ref: str | None = None
    ) -> None:
        """
        Clone a git repository to a target directory.

        Parameters
        ----------
        source:
            Source URL or local git path.
        target_dir:
            Clone destination.
        ref:
            Optional git reference to checkout.
        """
        self._run_command(['git', 'clone', source, str(target_dir)])
        if ref is not None:
            self._run_command(
                ['git', 'checkout', ref],
                cwd=target_dir,
            )

    def _current_commit(self, repo_dir: Path) -> str:
        """
        Return the current git commit hash.

        Parameters
        ----------
        repo_dir:
            Repository directory.

        Returns
        -------
        str
            Current commit hash.
        """
        return self._run_command(['git', 'rev-parse', 'HEAD'], cwd=repo_dir)

    def _current_ref(self, repo_dir: Path) -> str | None:
        """
        Return the current git branch name, if attached.

        Parameters
        ----------
        repo_dir:
            Repository directory.

        Returns
        -------
        str | None
            Current branch name or ``None`` for detached HEAD.
        """
        ref = self._run_command(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_dir,
        )
        return None if ref == 'HEAD' else ref

    def _update_repo(self, repo_dir: Path, ref: str | None = None) -> str:
        """
        Update a cloned git repository in place.

        Parameters
        ----------
        repo_dir:
            Repository directory.
        ref:
            Optional reference to track after fetching.

        Returns
        -------
        str
            Current commit hash after the update.
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
        Build a channel record from a checked-out repository.

        Parameters
        ----------
        local_name:
            Local channel alias.
        source:
            Original source string.
        repo_dir:
            Channel repository checkout.
        registered_at:
            Existing registration timestamp, if any.
        ref:
            Tracking ref override.

        Returns
        -------
        ChannelRecord
            Channel record ready to persist.
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
        Yield built-in skill directories and manifests.

        Yields
        ------
        tuple[Path, SkillManifest]
            Built-in skill directory and manifest.
        """
        if not self._builtin_dir.is_dir():
            return
        for child in sorted(self._builtin_dir.iterdir()):
            manifest_path = child / 'hiperhealth.yaml'
            if manifest_path.exists():
                yield child, self._read_skill_manifest_file(manifest_path)

    def _builtin_skill_name(
        self, manifest: SkillManifest, skill_dir: Path
    ) -> str:
        prefix = f'{BUILTIN_CHANNEL}.'
        if manifest.name.startswith(prefix):
            return manifest.name[len(prefix) :]
        return skill_dir.name

    def _iter_channel_skill_entries(
        self, local_name: str
    ) -> list[_ResolvedChannelSkill]:
        """
        Resolve all declared skills for one registered channel.

        Parameters
        ----------
        local_name:
            Local channel alias.

        Returns
        -------
        list[_ResolvedChannelSkill]
            Resolved available skills.
        """
        repo_dir = self._channel_repo_dir(local_name)
        if not repo_dir.is_dir():
            msg = f'Channel {local_name!r} is not registered.'
            raise KeyError(msg)

        channel_manifest = self._read_channel_manifest(repo_dir)
        resolved: list[_ResolvedChannelSkill] = []
        for declared in channel_manifest.skills:
            manifest_path = repo_dir / declared.manifest
            manifest = self._read_skill_manifest_file(manifest_path)
            resolved.append(
                _ResolvedChannelSkill(
                    available=AvailableSkillRecord(
                        channel=local_name,
                        name=declared.name,
                        canonical_id=_canonical_skill_id(
                            local_name, declared.name
                        ),
                        path=declared.path,
                        manifest_path=declared.manifest,
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
        Resolve an available channel skill from its canonical identifier.

        Parameters
        ----------
        skill_id:
            Canonical skill identifier.

        Returns
        -------
        _ResolvedChannelSkill | None
            Matching skill entry, if present.
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
        Install Python dependencies declared by a skill.

        Parameters
        ----------
        dependencies:
            Dependency specifiers to install with pip.
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
        Rewrite loaded skill metadata to its registry identifier.

        Parameters
        ----------
        skill:
            Instantiated skill.
        canonical_id:
            Registry identifier used to load the skill.

        Returns
        -------
        BaseSkill
            Skill instance with normalized metadata.
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
        Load an installed channel-based skill.

        Parameters
        ----------
        record:
            Installed skill record.

        Returns
        -------
        BaseSkill
            Instantiated skill.
        """
        manifest_path = Path(record.manifest_path)
        manifest = self._read_skill_manifest_file(manifest_path)
        skill_dir = manifest_path.parent
        cls = _load_class_from_directory(skill_dir, manifest.entry_point)
        skill = cls()
        return self._normalize_loaded_skill(skill, record.id)

    def _load_legacy_skill(self, record: InstalledSkillRecord) -> BaseSkill:
        """
        Load an installed legacy single-skill project.

        Parameters
        ----------
        record:
            Installed skill record.

        Returns
        -------
        BaseSkill
            Instantiated skill.
        """
        manifest_path = Path(record.manifest_path)
        manifest = self._read_skill_manifest_file(manifest_path)
        cls = _load_class_from_directory(
            manifest_path.parent, manifest.entry_point
        )
        skill = cls()
        return self._normalize_loaded_skill(skill, record.id)

    def add_channel(
        self,
        source: str,
        local_name: str | None = None,
        ref: str | None = None,
    ) -> str:
        """
        Register a channel from a git repository.

        Parameters
        ----------
        source:
            Git URL or local git repository path.
        local_name:
            Optional local alias for the channel.
        ref:
            Optional git reference to checkout.

        Returns
        -------
        str
            Final local channel alias.
        """
        if not self._looks_like_git_source(source):
            msg = (
                f'Cannot register channel from {source!r}. Provide a git URL '
                'or a local git repository path.'
            )
            raise ValueError(msg)

        self._ensure_storage_dirs()
        state = self._load_state()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_repo = Path(tmp_dir) / 'repo'
            self._clone_repo(source, temp_repo, ref=ref)
            source_kind = self._detect_source_kind(temp_repo)
            if source_kind != 'channel':
                msg = (
                    'The provided source is a legacy single-skill repository. '
                    'Use install() for backward-compatible single-skill '
                    'installs.'
                )
                raise ValueError(msg)

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
            source,
            target_repo,
            ref=ref,
        )
        state.channels[resolved_name] = record
        self._save_state(state)
        self._save_channel_metadata(resolved_name, record, target_repo)
        return resolved_name

    def list_channels(self) -> list[ChannelRecord]:
        """
        List registered channels.

        Returns
        -------
        list[ChannelRecord]
            Registered channels sorted by local alias.
        """
        state = self._load_state()
        return [state.channels[name] for name in sorted(state.channels.keys())]

    def list_channel_skills(
        self, local_name: str
    ) -> list[AvailableSkillRecord]:
        """
        List all skills declared by one registered channel.

        Parameters
        ----------
        local_name:
            Local channel alias.

        Returns
        -------
        list[AvailableSkillRecord]
            Declared channel skills.
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
        Update a registered channel checkout and refresh installed skills.

        Parameters
        ----------
        local_name:
            Local channel alias.
        ref:
            Optional git reference override.

        Returns
        -------
        list[str]
            Installed skill identifiers refreshed by the update.
        """
        state = self._load_state()
        channel = state.channels.get(local_name)
        if channel is None:
            msg = f'Channel {local_name!r} is not registered.'
            raise KeyError(msg)

        target_ref = ref if ref is not None else channel.ref
        repo_dir = self._channel_repo_dir(local_name)
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
                if record.artifact_path is not None:
                    shutil.rmtree(record.artifact_path, ignore_errors=True)
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
                source=channel.source,
            )
            updated.append(skill_id)

        self._save_state(state)
        self._save_channel_metadata(local_name, refreshed_channel, repo_dir)
        return sorted(updated)

    def remove_channel(self, local_name: str) -> None:
        """
        Remove a registered channel and its installed skills.

        Parameters
        ----------
        local_name:
            Local channel alias.
        """
        state = self._load_state()
        if local_name not in state.channels:
            msg = f'Channel {local_name!r} is not registered.'
            raise KeyError(msg)

        for skill_id, record in list(state.skills.items()):
            if record.channel != local_name:
                continue
            if record.artifact_path is not None:
                shutil.rmtree(record.artifact_path, ignore_errors=True)
            state.skills.pop(skill_id, None)

        shutil.rmtree(self._channel_dir(local_name), ignore_errors=True)
        state.channels.pop(local_name, None)
        self._save_state(state)

    def install(self, source: str) -> str:
        """
        Install a legacy single-skill project from a path or git URL.

        Parameters
        ----------
        source:
            Local path or git URL for a legacy single-skill project.

        Returns
        -------
        str
            Installed skill identifier.
        """
        source_path = Path(source)
        if source_path.is_dir():
            return self._install_legacy_from_path(source_path, source)

        if self._looks_like_git_source(source):
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_repo = Path(tmp_dir) / 'repo'
                self._clone_repo(source, temp_repo)
                return self._install_legacy_from_path(temp_repo, source)

        msg = (
            f'Cannot install from {source!r}. Provide a local directory path '
            'or a Git URL.'
        )
        raise ValueError(msg)

    def _install_legacy_from_path(
        self, source_path: Path, source_str: str
    ) -> str:
        """
        Install a legacy single-skill project from a directory.

        Parameters
        ----------
        source_path:
            Source directory.
        source_str:
            Original source string.

        Returns
        -------
        str
            Installed skill identifier.
        """
        if (
            not (source_path / 'skills.yaml').exists()
            and not (source_path / 'hiperhealth.yaml').exists()
        ):
            msg = (
                f'No hiperhealth.yaml found in {source_path}. Every skill '
                'project must include a hiperhealth.yaml metadata file.'
            )
            raise FileNotFoundError(msg)

        source_kind = self._detect_source_kind(source_path)
        if source_kind != 'legacy':
            msg = (
                'The provided source is a channel repository. '
                'Use add_channel() '
                'and install_skill() or install_channel() instead.'
            )
            raise ValueError(msg)

        manifest = self._read_manifest(source_path)
        state = self._load_state()
        existing = state.skills.get(manifest.name)
        if existing is not None and not existing.legacy:
            msg = (
                f'Skill id {manifest.name!r} already belongs to a registered '
                'channel install.'
            )
            raise ValueError(msg)

        self._ensure_storage_dirs()
        target_dir = self._artifact_dir(manifest.name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_path, target_dir)

        installed_at = existing.installed_at if existing else _utcnow()
        state.skills[manifest.name] = InstalledSkillRecord(
            id=manifest.name,
            skill_name=manifest.name,
            manifest_path=str(target_dir / 'hiperhealth.yaml'),
            installed_at=installed_at,
            updated_at=_utcnow(),
            version=manifest.version,
            source=source_str,
            artifact_path=str(target_dir),
            legacy=True,
        )
        self._save_state(state)
        self._install_dependencies(manifest.dependencies)
        return manifest.name

    def uninstall(self, name: str) -> None:
        """
        Backward-compatible alias for ``remove_skill()``.

        Parameters
        ----------
        name:
            Installed skill identifier.
        """
        self.remove_skill(name)

    def list_skills(
        self,
        channel: str | None = None,
        installed_only: bool = False,
    ) -> list[SkillSummary]:
        """
        List skills across built-ins, channels, and legacy installs.

        Parameters
        ----------
        channel:
            Optional local channel alias filter.
        installed_only:
            When ``True``, only include installed skills and built-ins.

        Returns
        -------
        list[SkillSummary]
            Skill summaries sorted by canonical identifier.
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
                        manifest_path=str(skill_dir / 'hiperhealth.yaml'),
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

        for record in state.skills.values():
            if not record.legacy:
                continue
            if channel is not None:
                continue
            manifest = self._read_skill_manifest_file(
                Path(record.manifest_path)
            )
            summaries.append(
                SkillSummary(
                    **manifest.model_dump(),
                    channel=None,
                    skill_name=record.skill_name,
                    canonical_id=record.id,
                    manifest_path=record.manifest_path,
                    installed=True,
                    enabled=record.enabled,
                )
            )

        unique: dict[str, SkillSummary] = {
            summary.canonical_id: summary for summary in summaries
        }
        return [unique[key] for key in sorted(unique.keys())]

    def install_skill(self, skill_id: str) -> str:
        """
        Install one skill from a registered channel.

        Parameters
        ----------
        skill_id:
            Canonical skill identifier.

        Returns
        -------
        str
            Installed canonical skill identifier.
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
        if existing is not None and existing.legacy:
            msg = (
                f'Skill id {skill_id!r} is already used by a legacy single-'
                'skill install.'
            )
            raise ValueError(msg)

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
            source=channel.source,
        )
        self._save_state(state)
        self._install_dependencies(available.manifest.dependencies)
        return skill_id

    def install_channel(
        self, local_name: str, include_disabled: bool = False
    ) -> list[str]:
        """
        Install all skills from a registered channel.

        Parameters
        ----------
        local_name:
            Local channel alias.
        include_disabled:
            Whether to include disabled skills.

        Returns
        -------
        list[str]
            Installed canonical skill identifiers.
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
        Refresh one installed skill.

        Parameters
        ----------
        skill_id:
            Canonical installed skill identifier.
        pull_channel:
            Whether to pull the channel checkout first.

        Returns
        -------
        str
            Refreshed skill identifier.
        """
        state = self._load_state()
        record = state.skills.get(skill_id)
        if record is None:
            msg = f'Skill {skill_id!r} is not installed.'
            raise KeyError(msg)

        if record.legacy:
            if not record.source:
                msg = f'Legacy skill {skill_id!r} has no update source.'
                raise ValueError(msg)
            return self.install(record.source)

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
            source=channel.source,
        )
        self._save_state(state)
        self._install_dependencies(available.manifest.dependencies)
        return skill_id

    def remove_skill(self, skill_id: str) -> None:
        """
        Remove one installed skill.

        Parameters
        ----------
        skill_id:
            Installed skill identifier.
        """
        if skill_id.startswith(f'{BUILTIN_CHANNEL}.'):
            msg = 'Built-in hiperhealth skills cannot be removed.'
            raise ValueError(msg)

        state = self._load_state()
        record = state.skills.get(skill_id)
        if record is None:
            msg = f'Skill {skill_id!r} is not installed.'
            raise KeyError(msg)

        if record.artifact_path is not None:
            shutil.rmtree(record.artifact_path, ignore_errors=True)
        state.skills.pop(skill_id, None)
        self._save_state(state)

    def load(self, name: str) -> BaseSkill:
        """
        Instantiate a skill by its canonical identifier.

        Parameters
        ----------
        name:
            Built-in canonical identifier, installed channel skill
            identifier, or installed legacy skill name.

        Returns
        -------
        BaseSkill
            Instantiated skill.
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
            if record.legacy:
                return self._load_legacy_skill(record)
            return self._load_channel_skill(record)

        if self._find_available_channel_skill(name) is not None:
            msg = (
                f'Skill {name!r} is available but not installed. Use '
                f'install_skill({name!r}) first.'
            )
            raise KeyError(msg)

        msg = (
            f'Skill {name!r} not found. Use list_skills() to inspect '
            'available skills and install_skill() or install() to add them.'
        )
        raise KeyError(msg)

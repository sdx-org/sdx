"""
title: Pipeline package — skill-based stage execution engine.
"""

from hiperhealth.pipeline.context import AuditEntry, PipelineContext
from hiperhealth.pipeline.discovery import discover_skills
from hiperhealth.pipeline.registry import SkillManifest, SkillRegistry
from hiperhealth.pipeline.runner import StageRunner
from hiperhealth.pipeline.skill import BaseSkill, Skill, SkillMetadata
from hiperhealth.pipeline.stages import Stage

__all__ = [
    'AuditEntry',
    'BaseSkill',
    'PipelineContext',
    'Skill',
    'SkillManifest',
    'SkillMetadata',
    'SkillRegistry',
    'Stage',
    'StageRunner',
    'create_default_runner',
    'discover_skills',
]


def create_default_runner() -> StageRunner:
    """
    title: Create a StageRunner with all built-in skills pre-configured.
    summary: |-
      Uses the SkillRegistry to load and register built-in skills
      in the standard order: privacy, extraction, diagnostics.
    returns:
      type: StageRunner
    """
    registry = SkillRegistry()
    runner = StageRunner(registry=registry)
    runner.register('hiperhealth.privacy')
    runner.register('hiperhealth.extraction')
    runner.register('hiperhealth.diagnostics')
    return runner

"""
title: Pipeline package — skill-based stage execution engine.
"""

from hiperhealth.pipeline.context import AuditEntry, PipelineContext
from hiperhealth.pipeline.discovery import discover_skills
from hiperhealth.pipeline.runner import StageRunner
from hiperhealth.pipeline.skill import BaseSkill, Skill, SkillMetadata
from hiperhealth.pipeline.stages import Stage

__all__ = [
    'AuditEntry',
    'BaseSkill',
    'PipelineContext',
    'Skill',
    'SkillMetadata',
    'Stage',
    'StageRunner',
    'create_default_runner',
    'discover_skills',
]


def create_default_runner() -> StageRunner:
    """
    title: Create a StageRunner with all built-in skills pre-configured.
    returns:
      type: StageRunner
    """
    from hiperhealth.skills.diagnostics import DiagnosticsSkill
    from hiperhealth.skills.extraction import ExtractionSkill
    from hiperhealth.skills.privacy import PrivacySkill

    return StageRunner(
        skills=[
            PrivacySkill(),
            ExtractionSkill(),
            DiagnosticsSkill(),
        ],
    )

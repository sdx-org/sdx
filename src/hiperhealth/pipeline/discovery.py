"""
title: Skill discovery via Python entry points.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from hiperhealth.pipeline.skill import BaseSkill


def discover_skills(
    group: str = 'hiperhealth.skills',
) -> list[BaseSkill]:
    """
    title: Load all installed skill classes and instantiate them.
    summary: |-
      Scans the ``hiperhealth.skills`` entry-point group for
      pip-installed skill packages.
    parameters:
      group:
        type: str
    returns:
      type: list[BaseSkill]
    """
    skills: list[BaseSkill] = []
    eps = entry_points(group=group)
    for ep in eps:
        skill_cls = ep.load()
        skills.append(skill_cls())
    return skills

"""
title: Skill app manifest discovery for notebook renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hiperhealth.pipeline import SkillRegistry, SkillUIView

if TYPE_CHECKING:
    from hiperhealth.pipeline import StageRunner


@dataclass(frozen=True)
class SkillAppViewRecord:
    """
    title: A resolved skill UI view ready for notebook rendering.
    attributes:
      skill_id:
        type: str
      skill_title:
        type: str
      view:
        type: SkillUIView
    """

    skill_id: str
    skill_title: str
    view: SkillUIView


def collect_skill_app_views(
    runner: StageRunner,
    *,
    stage: str | None = None,
    phase: str | None = None,
) -> list[SkillAppViewRecord]:
    """
    title: Return active skill app views filtered by stage and phase.
    parameters:
      runner:
        type: StageRunner
      stage:
        type: str | None
      phase:
        type: str | None
    returns:
      type: list[SkillAppViewRecord]
    """
    registry = runner.registry or SkillRegistry()
    skill_summaries = registry.list_skills()
    summaries = {summary.canonical_id: summary for summary in skill_summaries}
    summaries.update({summary.name: summary for summary in skill_summaries})
    records: list[SkillAppViewRecord] = []
    for skill in runner.skills:
        summary = summaries.get(skill.metadata.name)
        if summary is None or summary.app is None:
            continue
        title = summary.app.title or summary.description or summary.name
        for view in summary.app.views:
            if (
                stage is not None
                and not view.global_view
                and view.stage != stage
            ):
                continue
            if phase is not None and view.phase != phase:
                continue
            records.append(
                SkillAppViewRecord(
                    skill_id=skill.metadata.name,
                    skill_title=title,
                    view=view,
                )
            )
    return sorted(
        records,
        key=lambda record: (
            record.view.priority,
            record.skill_id,
            record.view.id,
        ),
    )

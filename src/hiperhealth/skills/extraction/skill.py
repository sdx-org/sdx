"""
title: ExtractionSkill — wraps medical report and wearable extractors.
"""

from __future__ import annotations

from typing import Any

from hiperhealth.pipeline.context import PipelineContext
from hiperhealth.pipeline.skill import BaseSkill, SkillMetadata
from hiperhealth.pipeline.stages import Stage
from hiperhealth.skills.extraction.medical_reports import (
    MedicalReportFileExtractor,
)
from hiperhealth.skills.extraction.wearable import (
    WearableDataFileExtractor,
)


class ExtractionSkill(BaseSkill):
    """
    title: Extracts text from medical reports and wearable data files.
    summary: |-
      Looks for file sources in ``ctx.extras['extraction_sources']``
      and stores extracted data in ``ctx.results['intake']``.
    attributes:
      _report_extractor:
        description: Value for _report_extractor.
      _wearable_extractor:
        description: Value for _wearable_extractor.
    """

    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='hiperhealth.extraction',
                version='0.4.0',
                stages=(Stage.INTAKE,),
                description=(
                    'Extract text from medical reports and '
                    'wearable data files.'
                ),
            )
        )
        self._report_extractor = MedicalReportFileExtractor()
        self._wearable_extractor = WearableDataFileExtractor()

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Extract data from sources listed in ctx.extras.
        parameters:
          stage:
            type: str
            description: Value for stage.
          ctx:
            type: PipelineContext
            description: Value for ctx.
        returns:
          type: PipelineContext
          description: Return value.
        """
        if stage != Stage.INTAKE:
            return ctx

        sources = ctx.extras.get('extraction_sources', {})
        results: dict[str, Any] = ctx.results.get(Stage.INTAKE, {})

        report_files = sources.get('medical_reports', [])
        for source in report_files:
            report = self._report_extractor.extract_report_data(source)
            results.setdefault('medical_reports', []).append(report)

        wearable_files = sources.get('wearable_data', [])
        for source in wearable_files:
            data = self._wearable_extractor.extract_wearable_data(source)
            results.setdefault('wearable_data', []).append(data)

        ctx.results[Stage.INTAKE] = results
        return ctx

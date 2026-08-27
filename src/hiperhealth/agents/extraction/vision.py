"""
title: Module for MedVision image extraction and analysis.
"""

from __future__ import annotations

import io

from pathlib import Path
from typing import Any, Optional, Union

from hiperhealth.llm import LLMSettings
from hiperhealth.vision.pipeline import (
    MedVisionPipeline,
    VisionExtractionResult,
)
from hiperhealth.vision.preprocessing import (
    ImageProcessor,
    ImageValidationError,
)

FileInput = Union[str, Path, io.BytesIO, bytes]


class MedicalVisionExtractorError(Exception):
    """
    title: Base exception for MedVision Extraction.
    """

    pass


class MedicalVisionExtractor:
    """
    title: Extracts structured clinical visual observations from images.
    attributes:
      processor:
        description: Handles image validation and standardization.
      pipeline:
        description: Executes the core visual analysis using LLM settings.
    """

    def __init__(self, llm_settings: Optional[LLMSettings] = None) -> None:
        """
        title: Initialize the MedicalVisionExtractor.
        parameters:
          llm_settings:
            type: Optional[LLMSettings]
            description: Optional LLMSettings overriding defaults.
        """
        self.processor = ImageProcessor()
        self.pipeline = MedVisionPipeline(settings=llm_settings)

    def extract_visual_features(
        self, source: FileInput, clinical_context: Optional[str] = None
    ) -> dict[str, Any]:
        """
        title: Validate, standardize, and analyze the image source.
        parameters:
          source:
            type: FileInput
            description: File path, BytesIO, or raw bytes containing the image.
          clinical_context:
            type: Optional[str]
            description: Optional text context to guide the LLM.
        returns:
          type: dict[str, Any]
          description: Dictionary containing the structured findings.
        """
        try:
            if isinstance(source, bytes):
                file_bytes = source
            elif isinstance(source, io.BytesIO):
                file_bytes = source.read()
                source.seek(0)
            elif isinstance(source, (str, Path)):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f'Image not found: {source}')
                file_bytes = path.read_bytes()
            else:
                raise ValueError('Unsupported source type for image analysis')

            self.processor.validate_integrity(file_bytes)

            standardized_img = self.processor.standardize(file_bytes)

            result: VisionExtractionResult = self.pipeline.analyze_image(
                standardized_img, clinical_context=clinical_context
            )
            return result.model_dump()

        except ImageValidationError as e:
            raise MedicalVisionExtractorError(
                f'Image preprocessing failed: {e}'
            ) from e
        except Exception as e:
            raise MedicalVisionExtractorError(
                f'Vision analysis failed: {e}'
            ) from e


def get_medical_vision_extractor() -> MedicalVisionExtractor:
    """
    title: Create and return an instance of MedicalVisionExtractor.
    returns:
      type: MedicalVisionExtractor
      description: An instance of MedicalVisionExtractor.
    """
    return MedicalVisionExtractor()

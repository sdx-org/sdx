from __future__ import annotations

import base64
import io

from PIL import Image
from pydantic import BaseModel

from hiperhealth.llm import LLMSettings, build_structured_llm


class VisionFinding(BaseModel):
    """
    title: Structured output from the Vision LLM.
    attributes:
      region_tag:
        type: str
      clinical_finding:
        type: str
      confidence_score:
        type: float
    """

    region_tag: str
    clinical_finding: str
    confidence_score: float


class VisionExtractionResult(BaseModel):
    """
    title: Collection of findings from a single image.
    attributes:
      findings:
        type: list[VisionFinding]
      image_quality_notes:
        type: str
    """

    findings: list[VisionFinding]
    image_quality_notes: str


class MedVisionPipeline:
    """
    title: Handles communication with Vision models.
    attributes:
      settings:
        description: LLM Settings
      llm:
        description: LLM instance
    """

    def __init__(self, settings: LLMSettings | None = None) -> None:
        """
        title: Initialize with LLMSettings.
        parameters:
          settings:
            type: LLMSettings | None
            description: Optional LLMSettings overriding defaults.
        """
        self.settings = settings or LLMSettings(
            provider='openai', model='gpt-4o-mini'
        )
        self.llm = build_structured_llm(settings=self.settings)

    def _image_to_base64(self, img: Image.Image) -> str:
        """
        title: Convert PIL Image to base64 string.
        parameters:
          img:
            type: Image.Image
            description: The PIL Image to convert.
        returns:
          type: str
          description: Base64 formatted string with data URI.
        """
        buffered = io.BytesIO()
        img.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{img_str}'

    def analyze_image(
        self, processed_image: Image.Image, clinical_context: str | None = None
    ) -> VisionExtractionResult:
        """
        title: Analyze standardized image, extract findings.
        parameters:
          processed_image:
            type: Image.Image
            description: Standardized clinical image input.
          clinical_context:
            type: str | None
            description: Optional context provided by the end user.
        returns:
          type: VisionExtractionResult
          description: The collection of findings returned by the model.
        """

        base64_img = self._image_to_base64(processed_image)
        schema_json = VisionExtractionResult.model_json_schema()

        system_prompt = (
            'You are an expert medical computer vision assistant. '
            'Analyze the provided clinical image for symptoms in eyes, '
            'nails, skin, tongue, or ears. '
            'Return a structured list of findings with confidence scores. '
            f'\n\nOutput must follow this JSON schema:\n{schema_json}'
        )

        user_prompt = 'Please analyze this clinical image.'
        if clinical_context:
            user_prompt += f' Context: {clinical_context}'

        from litellm import completion

        from hiperhealth.llm import (
            _coerce_model_output,
            _extract_message_content,
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': user_prompt},
                    {'type': 'image_url', 'image_url': {'url': base64_img}},
                ],
            },
        ]

        # Single optimized call
        kwargs = self.settings.to_litellm_kwargs()
        response = completion(
            messages=messages,
            response_format={'type': 'json_object'},
            **kwargs,
        )

        content = _extract_message_content(response)
        return _coerce_model_output(content, VisionExtractionResult)

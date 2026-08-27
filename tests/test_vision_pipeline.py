import io

from unittest.mock import MagicMock, patch

import pytest

from hiperhealth.agents.extraction.vision import (
    MedicalVisionExtractor,
    MedicalVisionExtractorError,
)
from hiperhealth.vision.pipeline import (
    MedVisionPipeline,
    VisionExtractionResult,
    VisionFinding,
)
from PIL import Image, ImageDraw


@pytest.fixture
def mock_image():
    """
    title: Mock Image
    """
    return Image.new('RGB', (224, 224), color='red')


@patch.dict('os.environ', {'OPENAI_API_KEY': 'fake-key'})
@patch('litellm.completion')
def test_pipeline_extraction(mock_completion, mock_image):
    """
    title: Test Pipeline Extraction
    parameters:
      mock_completion:
        description: Mock for litellm.completion.
      mock_image:
        description: Mock image for testing.
    """
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{"findings": [{"region_tag": "skin", '
        '"clinical_finding": "erythema", "confidence_score": 0.85}], '
        '"image_quality_notes": "Good"}'
    )
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    pipeline = MedVisionPipeline()
    result = pipeline.analyze_image(
        mock_image, clinical_context='Patient complains of rash'
    )

    assert isinstance(result, VisionExtractionResult)
    assert len(result.findings) == 1
    assert result.findings[0].region_tag == 'skin'
    assert result.findings[0].confidence_score == 0.85

    # Check that Litellm completion was called with image data
    _args, kwargs = mock_completion.call_args
    messages = kwargs.get('messages', [])
    assert len(messages) > 0
    assert 'image_url' in str(messages[1]['content'][1])


@patch('hiperhealth.agents.extraction.vision.MedVisionPipeline.analyze_image')
def test_medical_vision_extractor_flow(mock_analyze_image):
    """
    title: Test Medical Vision Extractor Flow
    parameters:
      mock_analyze_image:
        description: Mock for analyze_image.
    """
    # Create valid image bytes
    img = Image.new('RGB', (300, 300), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 0), (300, 300)], fill='black', width=2)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    valid_bytes = img_byte_arr.getvalue()

    # Mock pipeline returning standard result
    mock_analyze_image.return_value = VisionExtractionResult(
        findings=[
            VisionFinding(
                region_tag='eye',
                clinical_finding='clear',
                confidence_score=0.9,
            )
        ],
        image_quality_notes='OK',
    )

    extractor = MedicalVisionExtractor()
    result_dict = extractor.extract_visual_features(valid_bytes)

    assert 'findings' in result_dict
    assert result_dict['findings'][0]['region_tag'] == 'eye'


@patch('hiperhealth.agents.extraction.vision.MedVisionPipeline.analyze_image')
def test_medical_vision_extractor_input_types(mock_analyze_image, tmp_path):
    """
    title: Test Medical Vision Extractor Input Types
    parameters:
      mock_analyze_image:
        description: Mock for analyze_image.
      tmp_path:
        description: Temporary path for testing.
    """
    mock_analyze_image.return_value = VisionExtractionResult(
        findings=[
            VisionFinding(
                region_tag='eye',
                clinical_finding='clear',
                confidence_score=0.9,
            )
        ],
        image_quality_notes='OK',
    )

    img = Image.new('RGB', (300, 300), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 0), (300, 300)], fill='black', width=2)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    valid_bytes = img_byte_arr.getvalue()

    file_path = tmp_path / 'test_image.jpg'
    file_path.write_bytes(valid_bytes)

    extractor = MedicalVisionExtractor()

    # Test Path
    result = extractor.extract_visual_features(file_path)
    assert result['findings'][0]['region_tag'] == 'eye'

    # Test str path
    result = extractor.extract_visual_features(str(file_path))
    assert result['findings'][0]['region_tag'] == 'eye'

    # Test BytesIO
    result = extractor.extract_visual_features(io.BytesIO(valid_bytes))
    assert result['findings'][0]['region_tag'] == 'eye'

    # Test Unsupported type
    with pytest.raises(
        MedicalVisionExtractorError, match='Unsupported source type'
    ):
        extractor.extract_visual_features({'invalid': 'type'})


@patch('hiperhealth.agents.extraction.vision.MedVisionPipeline.analyze_image')
def test_medical_vision_extractor_pipeline_error(mock_analyze_image):
    """
    title: Test Medical Vision Extractor Pipeline Error
    parameters:
      mock_analyze_image:
        description: Mock for analyze_image.
    """
    mock_analyze_image.side_effect = Exception('API down')

    img = Image.new('RGB', (300, 300), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.line([(0, 0), (300, 300)], fill='black', width=2)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    valid_bytes = img_byte_arr.getvalue()

    extractor = MedicalVisionExtractor()
    with pytest.raises(
        MedicalVisionExtractorError, match='Vision analysis failed: API down'
    ):
        extractor.extract_visual_features(valid_bytes)

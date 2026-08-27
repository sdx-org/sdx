import io

import pytest

from hiperhealth.vision.preprocessing import (
    ImageProcessor,
    ImageValidationError,
)
from PIL import Image, ImageDraw, ImageFilter


@pytest.fixture
def processor():
    """
    title: Processor
    """
    return ImageProcessor()


@pytest.fixture
def valid_image_bytes():
    """
    title: Valid Image Bytes
    """
    # High-contrast sharp image with avg grey bg to pass exposure checks
    img = Image.new('RGB', (500, 500), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    # Adding sharp lines to ensure high Laplacian variance
    for i in range(0, 500, 50):
        draw.line([(i, 0), (i, 500)], fill='black', width=2)

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


@pytest.fixture
def blurry_image_bytes():
    """
    title: Blurry Image Bytes
    """
    # Create a sharp image then blur it heavily
    img = Image.new('RGB', (300, 300), color='red')
    img = img.filter(ImageFilter.GaussianBlur(radius=20))

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


# --- Tests ---


def test_validate_integrity_success(processor, valid_image_bytes):
    """
    title: Test Validate Integrity Success
    parameters:
      processor:
        description: ImageProcessor instance.
      valid_image_bytes:
        description: Valid image bytes.
    """
    assert processor.validate_integrity(valid_image_bytes) is True


def test_validate_integrity_fails_on_blur(processor, blurry_image_bytes):
    """
    title: Test Validate Integrity Fails on Blur
    parameters:
      processor:
        description: ImageProcessor instance.
      blurry_image_bytes:
        description: Blurry image bytes.
    """
    with pytest.raises(ImageValidationError, match='too blurry'):
        processor.validate_integrity(blurry_image_bytes)


def test_validate_integrity_fails_on_corrupt(processor):
    """
    title: Test Validate Integrity Fails on Corrupt
    parameters:
      processor:
        description: ImageProcessor instance.
    """
    bad_data = b'not_an_image_file_at_all'
    with pytest.raises(ImageValidationError):
        processor.validate_integrity(bad_data)


def test_standardize_logic(processor, valid_image_bytes):
    """
    title: Test Standardize Logic
    parameters:
      processor:
        description: ImageProcessor instance.
      valid_image_bytes:
        description: Valid image bytes.
    """
    standardized_img = processor.standardize(valid_image_bytes)

    assert isinstance(standardized_img, Image.Image)
    assert standardized_img.size == (224, 224)
    assert standardized_img.mode == 'RGB'


def test_standardize_converts_rgba(processor):
    """
    title: Test Standardize Converts RGBA
    parameters:
      processor:
        description: ImageProcessor instance.
    """
    rgba_img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    img_byte_arr = io.BytesIO()
    rgba_img.save(img_byte_arr, format='PNG')

    standardized_img = processor.standardize(img_byte_arr.getvalue())
    assert standardized_img.mode == 'RGB'


@pytest.fixture
def underexposed_image_bytes():
    """
    title: Underexposed Image Bytes
    """
    # Use 100x100 black image with 4 sharp white lines
    # This gives high Laplacian variance (sharpness)
    # But very low average brightness (4%)
    img = Image.new('RGB', (100, 100), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(10, 50, 10):
        draw.line([(i, 0), (i, 100)], fill=(255, 255, 255), width=1)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


@pytest.fixture
def overexposed_image_bytes():
    """
    title: Overexposed Image Bytes
    """
    # Use 100x100 white image with 4 sharp black lines
    # This gives high Laplacian variance
    # But very high average brightness (96%)
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(10, 50, 10):
        draw.line([(i, 0), (i, 100)], fill=(0, 0, 0), width=1)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def test_validate_integrity_fails_on_underexposed(
    processor, underexposed_image_bytes
):
    """
    title: Test Validate Integrity Fails on Underexposed
    parameters:
      processor:
        description: ImageProcessor instance.
      underexposed_image_bytes:
        description: Underexposed image bytes.
    """
    with pytest.raises(ImageValidationError, match='underexposed'):
        processor.validate_integrity(underexposed_image_bytes)


def test_validate_integrity_fails_on_overexposed(
    processor, overexposed_image_bytes
):
    """
    title: Test Validate Integrity Fails on Overexposed
    parameters:
      processor:
        description: ImageProcessor instance.
      overexposed_image_bytes:
        description: Overexposed image bytes.
    """
    with pytest.raises(ImageValidationError, match='overexposed'):
        processor.validate_integrity(overexposed_image_bytes)

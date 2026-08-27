"""
title: Image preprocessing utilities for MedVision pipeline.
"""

from __future__ import annotations

import io

from typing import ClassVar

import cv2
import numpy as np

from PIL import Image, ImageStat


class ImageValidationError(ValueError):
    """
    title: Raised when an image fails validation checks.
    """

    pass


class ImageProcessor:
    """
    title: >-
      Validates and standardizes medical images for the MedVision pipeline.
    attributes:
      ALLOWED_MIME_TYPES:
        type: ClassVar[set[str]]
    """

    # Constants for validation thresholds
    MIN_VAR_LAPLACIAN = 100.0  # Threshold for blur detection
    EXPOSURE_THRESHOLD_LOW = 0.05  # Reject if >95% pure black
    EXPOSURE_THRESHOLD_HIGH = 0.95  # Reject if >95% pure white
    TARGET_SIZE = (224, 224)
    ALLOWED_MIME_TYPES: ClassVar[set[str]] = {'image/jpeg', 'image/png'}

    @staticmethod
    def _get_mime_type(file_bytes: bytes) -> str:
        """
        title: Get the MIME type of a byte buffer using Pillow.
        parameters:
          file_bytes:
            type: bytes
            description: The file to identify.
        returns:
          type: str
          description: The best guessed MIME type string.
        """
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                fmt = img.format
                if fmt == 'JPEG':
                    return 'image/jpeg'
                elif fmt == 'PNG':
                    return 'image/png'
                elif fmt:
                    return f'image/{fmt.lower()}'
                return 'application/octet-stream'
        except Exception:
            return 'application/octet-stream'

    def validate_integrity(self, file_bytes: bytes) -> bool:
        """
        title: >-
          Check if the file is an image, is not corrupted, is not too blurry,
          and has acceptable exposure.
        parameters:
          file_bytes:
            type: bytes
            description: The raw bytes of the image file.
        returns:
          type: bool
          description: True if valid.
        """
        # 1. Security/MIME Check
        mime_type = self._get_mime_type(file_bytes)
        if mime_type not in self.ALLOWED_MIME_TYPES:
            raise ImageValidationError(
                f'Invalid MIME type: {mime_type}. '
                f'Allowed: {self.ALLOWED_MIME_TYPES}'
            )

        try:
            # 2. Corruption check using Pillow
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()  # Verifies the file is a valid image

            # Reopen because verify() closes the file or alters the pointer
            img = Image.open(io.BytesIO(file_bytes))

            # Convert to numpy array for OpenCV ops (needs RGB/Grayscale)
            cv_img = cv2.cvtColor(
                np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR
            )

            # 3. Exposure Check
            # Convert to grayscale to check brightness
            stat = ImageStat.Stat(img.convert('L'))
            avg_brightness = stat.mean[0] / 255.0

            if avg_brightness < self.EXPOSURE_THRESHOLD_LOW:
                raise ImageValidationError('Image is too dark (underexposed).')
            elif avg_brightness > self.EXPOSURE_THRESHOLD_HIGH:
                raise ImageValidationError(
                    'Image is too bright (overexposed).'
                )

            # 4. Blur Detection: Variance of Laplacian
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

            if variance_of_laplacian < self.MIN_VAR_LAPLACIAN:
                raise ImageValidationError(
                    'Image is too blurry for clinical use. '
                    f'Score: {variance_of_laplacian:.2f} '
                    f'< {self.MIN_VAR_LAPLACIAN}'
                )

        except Exception as e:
            if not isinstance(e, ImageValidationError):
                raise ImageValidationError(
                    f'Corrupted or invalid image file: {e!s}'
                )
            raise e

        return True

    def standardize(self, file_bytes: bytes) -> Image.Image:
        """
        title: Standardizes the image for clinical ML model inference.
        parameters:
          file_bytes:
            type: bytes
            description: The raw bytes of the image file.
        returns:
          type: Image.Image
          description: The standardized PIL Image.
        """
        img: Image.Image = Image.open(io.BytesIO(file_bytes))

        # 1. Color Space Conversion
        if img.mode in ('RGBA', 'LA') or (
            img.mode == 'P' and 'transparency' in img.info
        ):
            # Create a white background and paste the image over it
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(
                img,
                mask=img.split()[3]
                if img.mode == 'RGBA'
                else img.convert('RGBA').split()[3],
            )
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # 2. Aspect-Ratio-Preserving Resize (Letterbox/Pad)
        w, h = img.size
        scale = min(self.TARGET_SIZE[0] / w, self.TARGET_SIZE[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Use high quality downsampling
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new(
            'RGB', self.TARGET_SIZE, (0, 0, 0)
        )  # Pad with zero/black

        paste_x = (self.TARGET_SIZE[0] - new_w) // 2
        paste_y = (self.TARGET_SIZE[1] - new_h) // 2
        new_img.paste(img_resized, (paste_x, paste_y))

        return new_img

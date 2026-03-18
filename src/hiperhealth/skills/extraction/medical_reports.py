"""
title: Module for extracting text from PDF documents and images.
"""

from __future__ import annotations

import io

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    IO,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    cast,
)

import magic
import pytesseract

from PIL import Image
from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfStreamError


class MedicalReportExtractorError(Exception):
    """
    title: Base class for Medical Report Extraction from pdf/images.
    """

    ...


class TextExtractionError(MedicalReportExtractorError):
    """
    title: Exception raised for errors in text extraction.
    """

    ...


T = TypeVar('T')


FileInput = str | Path | IO[bytes] | io.BytesIO
FileExtension = Literal['pdf', 'png', 'jpg', 'jpeg']
MimeType = Literal['application/pdf', 'image/png', 'image/jpeg']


class BaseMedicalReportExtractor(ABC, Generic[T]):
    """
    title: Base class for medical report extraction.
    """

    @abstractmethod
    def extract_report_data(
        self,
        source: T,
    ) -> dict[str, Any]:
        """
        title: Extract structured text data from source file.
        parameters:
          source:
            type: T
            description: Value for source.
        returns:
          type: dict[str, Any]
          description: Return value.
        """
        raise NotImplementedError


class MedicalReportFileExtractor(BaseMedicalReportExtractor[FileInput]):
    """
    title: Extract medical report text from files and in-memory objects.
    attributes:
      allowed_extensions_mimetypes_map:
        type: ClassVar[dict[FileExtension, MimeType]]
        description: Value for allowed_extensions_mimetypes_map.
      _mimetype_cache:
        type: dict[str, MimeType]
        description: Value for _mimetype_cache.
      _text_cache:
        type: dict[str, str]
        description: Value for _text_cache.
      mime:
        description: Value for mime.
    """

    allowed_extensions_mimetypes_map: ClassVar[
        dict[FileExtension, MimeType]
    ] = {
        'pdf': 'application/pdf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
    }

    def __init__(self) -> None:
        """
        title: Initialize extractor with caches and mimetype detector.
        """
        self._mimetype_cache: dict[str, MimeType] = {}
        self._text_cache: dict[str, str] = {}
        self.mime = magic.Magic(mime=True)

    @property
    def allowed_extensions(self) -> list[FileExtension]:
        """
        title: Return supported file extensions.
        returns:
          type: list[FileExtension]
          description: Return value.
        """
        return list(self.allowed_extensions_mimetypes_map.keys())

    @property
    def allowed_mimetypes(self) -> list[MimeType]:
        """
        title: Return supported MIME types.
        returns:
          type: list[MimeType]
          description: Return value.
        """
        return list(self.allowed_extensions_mimetypes_map.values())

    def extract_report_data(
        self,
        source: FileInput,
    ) -> dict[str, Any]:
        """
        title: Validate input and return extracted text plus basic metadata.
        parameters:
          source:
            type: FileInput
            description: Value for source.
        returns:
          type: dict[str, Any]
          description: Return value.
        """
        self._validate_or_raise(source)
        return self._process_file(source)

    def _validate_or_raise(self, source: FileInput) -> None:
        """
        title: Check existence, type support, and non-empty streams.
        parameters:
          source:
            type: FileInput
            description: Value for source.
        """
        if isinstance(source, io.BytesIO):
            data = source.read(10)
            source.seek(0)
            if not data:
                raise FileNotFoundError('In-memory file is empty')
        elif isinstance(source, (str, Path)):
            if not Path(source).exists():
                raise FileNotFoundError(f'File not found: {source}')

        mime = self._get_mime_type(source)
        if mime not in self.allowed_mimetypes:
            raise MedicalReportExtractorError(f'Unsupported MIME type: {mime}')

    def _process_file(
        self,
        source: FileInput,
    ) -> dict[str, Any]:
        """
        title: Extract text and normalize the result payload.
        parameters:
          source:
            type: FileInput
            description: Value for source.
        returns:
          type: dict[str, Any]
          description: Return value.
        """
        mime = self._get_mime_type(source)
        text = self._extract_text(source)
        return self._build_report_payload(source, text, mime)

    def _get_cache_key(self, source: FileInput) -> str:
        """
        title: Return cache key for the source object.
        parameters:
          source:
            type: FileInput
            description: Value for source.
        returns:
          type: str
          description: Return value.
        """
        if isinstance(source, (Path, str)):
            return str(Path(source).resolve())
        return str(id(source))

    def _get_mime_type(self, source: FileInput) -> MimeType:
        """
        title: Detect MIME type and cache for the source.
        parameters:
          source:
            type: FileInput
            description: Value for source.
        returns:
          type: MimeType
          description: Return value.
        """
        key = self._get_cache_key(source)
        if key in self._mimetype_cache:
            return self._mimetype_cache[key]

        if isinstance(source, (Path, str)):
            mime = self.mime.from_file(str(source))
        else:
            head = source.read(2048)
            source.seek(0)
            mime = self.mime.from_buffer(head)

        # Cast mime string to Literal MIME type for type safety
        mime_literal = cast(MimeType, mime)
        self._mimetype_cache[key] = mime_literal
        return mime_literal

    def _extract_text(self, source: FileInput) -> str:
        """
        title: Extract cached raw text from source.
        parameters:
          source:
            type: FileInput
            description: Value for source.
        returns:
          type: str
          description: Return value.
        """
        key = self._get_cache_key(source)
        if key in self._text_cache:
            return self._text_cache[key]

        mime = self._get_mime_type(source)
        if mime == 'application/pdf':
            text = self._extract_text_from_pdf(source)
        else:
            text = self._extract_text_from_image(source)

        self._text_cache[key] = text
        return text

    def extract_text(self, source: FileInput) -> str:
        """
        title: Validate input and return the extracted raw text only.
        parameters:
          source:
            type: FileInput
            description: Value for source.
        returns:
          type: str
          description: Return value.
        """
        self._validate_or_raise(source)
        return self._extract_text(source)

    def _extract_text_from_pdf(self, pdf_source: FileInput) -> str:
        """
        title: Extract text content from a PDF file or in-memory stream.
        parameters:
          pdf_source:
            type: FileInput
            description: Value for pdf_source.
        returns:
          type: str
          description: Return value.
        """
        try:
            if isinstance(pdf_source, io.BytesIO):
                data = pdf_source.read()
                pdf_source.seek(0)
                reader = PdfReader(io.BytesIO(data))
            else:
                reader = PdfReader(pdf_source)
        except (PdfStreamError, EmptyFileError) as e:
            raise TextExtractionError(f'Failed to parse PDF: {e}') from e

        text_pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)

        if not text_pages:
            raise TextExtractionError('No extractable text in PDF')

        return '\n'.join(text_pages)

    def _extract_text_from_image(self, img_source: FileInput) -> str:
        """
        title: Extract text from images using OCR.
        parameters:
          img_source:
            type: FileInput
            description: Value for img_source.
        returns:
          type: str
          description: Return value.
        """
        if isinstance(img_source, (str, Path)):
            img = Image.open(img_source)
        else:
            data = io.BytesIO(img_source.read())
            img = Image.open(data)

        text: str = pytesseract.image_to_string(img)
        if not text.strip():
            raise TextExtractionError('No extractable text in image')
        return text

    def _build_report_payload(
        self,
        source: FileInput,
        text: str,
        mime: MimeType,
    ) -> dict[str, Any]:
        """
        title: Build a stable structured payload around locally extracted text.
        parameters:
          source:
            type: FileInput
            description: Value for source.
          text:
            type: str
            description: Value for text.
          mime:
            type: MimeType
            description: Value for mime.
        returns:
          type: dict[str, Any]
          description: Return value.
        """
        source_type = 'pdf' if mime == 'application/pdf' else 'image'
        source_name = (
            Path(source).name
            if isinstance(source, (str, Path))
            else 'in_memory'
        )
        return {
            'source_name': source_name,
            'source_type': source_type,
            'mime_type': mime,
            'text': text,
        }


def get_medical_report_extractor() -> MedicalReportFileExtractor:
    """
    title: Create and return an instance of MedicalReportFileExtractor.
    returns:
      type: MedicalReportFileExtractor
      description: Return value.
    """
    return MedicalReportFileExtractor()

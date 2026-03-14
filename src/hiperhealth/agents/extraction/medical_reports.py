"""Module for extracting FHIR data from PDF documents and images."""

from __future__ import annotations

import io

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    IO,
    Any,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
)

import magic
import pytesseract

from anamnesisai import AnamnesisAI
from PIL import Image
from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfStreamError

from hiperhealth.llm import LLMSettings, load_fhir_llm_settings
from hiperhealth.utils import make_json_serializable


# Exceptions
class MedicalReportExtractorError(Exception):
    """Base class for Medical Report Extraction from pdf/images."""

    ...


class TextExtractionError(MedicalReportExtractorError):
    """Exception raised for errors in text extraction."""

    ...


class FHIRConversionError(MedicalReportExtractorError):
    """Exception raised for errors during FHIR conversion."""

    ...


# TypeVars for generics
T = TypeVar('T')


# Types
FileInput = Union[str, Path, IO[bytes], io.BytesIO]
FileExtension = Literal['pdf', 'png', 'jpg', 'jpeg']
MimeType = Literal['application/pdf', 'image/png', 'image/jpeg']


class BaseMedicalReportExtractor(ABC, Generic[T]):
    """Base class for medical report extraction."""

    @abstractmethod
    def extract_report_data(
        self,
        source: T,
        api_key: Optional[str] = None,
        *,
        llm_settings: LLMSettings | None = None,
    ) -> Dict[str, Any]:
        """Extract structured report data from source file."""
        raise NotImplementedError


class MedicalReportFileExtractor(BaseMedicalReportExtractor[FileInput]):
    """Extract medical report data from files and in-memory objects."""

    allowed_extensions_mimetypes_map: ClassVar[
        Dict[FileExtension, MimeType]
    ] = {
        'pdf': 'application/pdf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
    }

    def __init__(
        self,
        *,
        llm_settings: LLMSettings | None = None,
        anamnesis_factory: Any = AnamnesisAI,
    ) -> None:
        """Initialize extractor with caches and mimetype detector."""
        self._mimetype_cache: Dict[str, MimeType] = {}
        self._text_cache: Dict[str, str] = {}
        self.mime = magic.Magic(mime=True)
        self.llm_settings = llm_settings
        self._anamnesis_factory = anamnesis_factory

    @property
    def allowed_extensions(self) -> List[FileExtension]:
        """Return supported file extensions."""
        return list(self.allowed_extensions_mimetypes_map.keys())

    @property
    def allowed_mimetypes(self) -> List[MimeType]:
        """Return supported MIME types."""
        return list(self.allowed_extensions_mimetypes_map.values())

    def extract_report_data(
        self,
        source: FileInput,
        api_key: Optional[str] = None,
        *,
        llm_settings: LLMSettings | None = None,
    ) -> Dict[str, Any]:
        """Validate and process input to extract FHIR-compliant report data."""
        self._validate_or_raise(source)
        return self._process_file(source, api_key, llm_settings=llm_settings)

    def _validate_or_raise(self, source: FileInput) -> None:
        """Check existence, type support, and non-empty streams."""
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
        api_key: Optional[str] = None,
        *,
        llm_settings: LLMSettings | None = None,
    ) -> Dict[str, Any]:
        """Extract text and convert to FHIR."""
        text = self._extract_text(source)
        return self._convert_to_fhir(
            text,
            api_key,
            llm_settings=llm_settings,
        )

    def _get_cache_key(self, source: FileInput) -> str:
        """Return cache key for the source object."""
        if isinstance(source, (Path, str)):
            return str(Path(source).resolve())
        return str(id(source))

    def _get_mime_type(self, source: FileInput) -> MimeType:
        """Detect MIME type and cache for the source."""
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
        """Extract cached raw text from source."""
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

    def _extract_text_from_pdf(self, pdf_source: FileInput) -> str:
        """Extract text content from a PDF file or in-memory stream."""
        try:
            if isinstance(pdf_source, io.BytesIO):
                data = pdf_source.read()
                pdf_source.seek(0)
                reader = PdfReader(io.BytesIO(data))
            else:
                reader = PdfReader(pdf_source)
        except (PdfStreamError, EmptyFileError) as e:
            raise TextExtractionError(f'Failed to parse PDF: {e}') from e

        text_pages: List[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)

        if not text_pages:
            raise TextExtractionError('No extractable text in PDF')

        return '\n'.join(text_pages)

    def _extract_text_from_image(self, img_source: FileInput) -> str:
        """Extract text from images using OCR."""
        if isinstance(img_source, (str, Path)):
            img = Image.open(img_source)
        else:
            data = io.BytesIO(img_source.read())
            img = Image.open(data)

        text: str = pytesseract.image_to_string(img)
        if not text.strip():
            raise TextExtractionError('No extractable text in image')
        return text

    def _convert_to_fhir(
        self,
        text_content: str,
        api_key: Optional[str] = None,
        *,
        llm_settings: LLMSettings | None = None,
    ) -> Dict[str, Any]:
        """Convert extracted text to FHIR resources using Anamnesisai."""
        settings = self._resolve_llm_settings(
            api_key=api_key,
            llm_settings=llm_settings,
        )
        backend = settings.to_anamnesis_backend()
        key = settings.api_key

        if backend == 'openai' and not key:
            raise EnvironmentError('Missing OpenAI API key')

        anaai = self._anamnesis_factory(
            backend=backend,
            api_key=key,
            api_params=settings.to_anamnesis_api_params(),
        )
        resources = anaai.extract_fhir(text_content)
        result: Dict[str, Any] = make_json_serializable(
            {res.__class__.__name__: res.model_dump() for res in resources[0]}
        )
        return result

    def _resolve_llm_settings(
        self,
        *,
        api_key: Optional[str] = None,
        llm_settings: LLMSettings | None = None,
    ) -> LLMSettings:
        """Resolve LLM settings from call overrides, constructor, and env."""
        settings = (
            llm_settings or self.llm_settings or load_fhir_llm_settings()
        )
        if api_key:
            settings = settings.with_overrides(api_key=api_key)
        return settings


def get_medical_report_extractor(
    *,
    llm_settings: LLMSettings | None = None,
) -> MedicalReportFileExtractor:
    """Create and return an instance of MedicalReportFileExtractor."""
    return MedicalReportFileExtractor(llm_settings=llm_settings)

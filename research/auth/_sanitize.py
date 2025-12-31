from pydantic import Field
from pydantic.dataclasses import dataclass
import re
import sys
import unicodedata

TERMINATE_ON_EXTRA_FAILURE: bool = True

@dataclass
class DetectorResult:
    entity: str = Field(..., description="The type of entity that was detected.")
    start: int = Field(..., description="The start index of the detected entity.")
    end: int = Field(..., description="The end index of the detected entity.")

class BaseDetector:
    """Base class for detectors."""

    def get_entities(
        self,
        results: list[DetectorResult]
    ) -> list[str]:
        """Returns a list of entities from a list of DetectorResult objects.

        Args:
            results: A list of DetectorResult objects.
        Returns:
            A list of entities.
        """
        return [result.entity for result in results]

    def detect_all(
        self,
        text: str,
        *args,
        **kwargs
    ) -> list[DetectorResult]:
        """Performs detection on the given text and returns a list of DetectorResult objects.

        Args:
            text: The text to analyze.
        Returns:
            A list of DetectorResult objects.
        """
        raise NotImplementedError("")

    def detect(
        self,
        text: str,
        *args,
        **kwargs
    ) -> bool:
        """Performs detection on the given text and returns a boolean indicating whether there has been any detection.

        Args:
            text: The text to analyze.
        Returns:
            A boolean indicating whether there has been any detection.
        """
        return len(self.detect_all(text, *args, **kwargs)) > 0

    async def preload(self):
        """
        Some workload to run to initialize the detector for lower-latency inference later on.

        For instance, model loading or other expensive operations.
        """
        pass

class ExtrasImport:

    def __init__(
        self,
        import_name,
        package_name,
        version_constraint
    ) -> None:
        """Creates a new ExtrasImport object.

        Args:
            import_name (str): The name or specifier of the module to import (e.g. 'lib' or 'lib.submodule')
            package_name (str): The name of the pypi package that contains the module.
            version_constraint (str): The version constraint for the package (e.g. '>=1.0.0')
        """
        self.name = import_name
        self.package_name = package_name
        self.version_constraint = version_constraint

        # collection of sites where this dependency is used
        # (only available if find_all is used)
        self.sites = []

    def import_names(
        self,
        *specifiers
    ):

        module = self.import_module()
        elements = [getattr(module, specifier) for specifier in specifiers]
        if len(elements) == 1:
            return elements[0]
        return elements

    def import_module(self):
        module = __import__(self.name, fromlist=[self.name])
        return module

    def __str__(self):
        if len(self.sites) > 0:
            sites_str = f", sites={self.sites}"
        else:
            sites_str = ""
        return f"ExtrasImport('{self.name}', '{self.package_name}', '{self.version_constraint}'{sites_str})"

    def __repr__(self):
        return str(self)



class Extra:
    """
    An Extra is a group of optional dependencies that can be installed on demand.
    The extra is defined by a name, a description, and a collection of packages.
    For a list of available extras, see `Extra.find_all()` and below.
    """

    def __init__(
        self,
        name,
        description,
        packages
    ):
        self.name = name
        self.description = description
        self.packages = packages
        self._is_available = None

        Extra.extras[name] = self

    def is_available(self) -> bool:
        """Returns whether the extra is available (all assigned imports can be resolved)."""
        if self._is_available is not None:
            return self._is_available

        for package in self.packages.values():
            try:
                __import__(package.name)
            except ImportError:
                self._is_available = False
                return False

        self._is_available = True
        return True

    def package(self, name) -> ExtrasImport:
        """Returns the package with the given name."""
        if not self.is_available():
            self.install()

        return self.packages[name]

    def install(self):
        """Installs all required packages for this extra (using pip if available)."""
        # like for imports, but all in one go
        msg = "warning: you are trying to use a feature that relies on the extra dependency '{}', which requires the following packages to be installed:\n".format(
            self.name
        )
        for imp in self.packages.values():
            msg += "   - " + imp.package_name + imp.version_constraint + "\n"

        sys.stderr.write(msg + "\n")

        # check if terminal input is possible
        if sys.stdin.isatty():
            sys.stderr.write("Press (y/enter) to install the packages or Ctrl+C to exit: ")
            answer = input()
            if answer == "y" or len(answer) == 0:
                import subprocess

                # check if 'pip' is installed
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "--version"], capture_output=True
                )
                if result.returncode != 0:
                    sys.stderr.write(
                        "error: 'pip' is not installed. Please install the above mentioned packages manually.\n"
                    )
                    if TERMINATE_ON_EXTRA_FAILURE:
                        sys.exit(1)
                    else:
                        raise RuntimeError(
                            "policy execution failed due to missing dependencies in the runtime environment"
                        )
                for imp in self.packages.values():
                    subprocess.call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            f"{imp.package_name}{imp.version_constraint}",
                        ]
                    )
            else:
                if TERMINATE_ON_EXTRA_FAILURE:
                    sys.exit(1)
                else:
                    raise RuntimeError(
                        "policy execution failed due to missing dependencies in the runtime environment"
                    )
        else:
            if TERMINATE_ON_EXTRA_FAILURE:
                sys.exit(1)
            else:
                raise RuntimeError(
                    "policy execution failed due to missing dependencies in the runtime environment"
                )

    @staticmethod
    def find_all() -> list["Extra"]:
        return list(Extra.extras.values())

class UnicodeDetector(BaseDetector):
    """
    Detector for detecting unicode characters based on their category (using allow or deny list).
    """

    def detect_all(self, text: str, categories: list[str] | None = None) -> list[DetectorResult]:
        """Detects all unicode groups that should not be allowed in the text.

        Attributes:
            allow: List of categories to allow.
            deny: List of categories to deny.

        Returns:
            A list of DetectorResult objects indicating the detected unicode groups.

        Raises:
            ValueError: If both allow and deny categories are specified.
        """
        res = []
        for index, chr in enumerate(text):
            cat = unicodedata.category(chr)
            if categories is None or cat in categories:
                res.append(DetectorResult(cat, index, index + 1))
        return res

_SECRETS_PATTERNS = {
    "GITHUB_TOKEN": [
        re.compile(r'(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36}'),
    ],
    "AWS_ACCESS_KEY": [
        re.compile(r'(?:A3T[A-Z0-9]|ABIA|ACCA|AKIA|ASIA)[0-9A-Z]{16}'),
        re.compile(r'aws.{{0,20}}?{secret_keyword}.{{0,20}}?[\'\"]\
                ([0-9a-zA-Z/+]{{40}})[\'\"]'.format(
            secret_keyword=r'(?:key|pwd|pw|password|pass|token)',
        ), flags=re.IGNORECASE),
    ],
    "AZURE_STORAGE_KEY": [
        re.compile(r'AccountKey=[a-zA-Z0-9+\/=]{88}'),
    ],
    "SLACK_TOKEN": [
        re.compile(r'xox(?:a|b|p|o|s|r)-(?:\d+-)+[a-z0-9]+', flags=re.IGNORECASE),
        re.compile(r'https://hooks\.slack\.com/services/T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+', flags=re.IGNORECASE | re.VERBOSE),
    ],
    "GOOGLE_API_KEY": [
        re.compile(r'\bAIza[0-9A-Za-z\-_]{35}\b'),
        re.compile(r'\bya29\.[0-9A-Za-z\-_\.]{40,}\b'),
    ],
    "GEMINI_aPI_KEY": [
        re.compile(r'\bAIza[0-9A-Za-z\-_]{35}\b'),
        re.compile(r'\bya29\.[0-9A-Za-z\-_\.]{40,}\b'),
        re.compile(r'\bgemini-[A-Za-z0-9_\-]{20,120}\b', flags=re.IGNORECASE),
    ],
    "OPENAI_API_KEY": [
        re.compile(r'\bsk-(?:live|test|proj)-[A-Za-z0-9_\-]{24,120}\b'),
        re.compile(r'\bsk-[A-Za-z0-9_\-]{24,120}\b'),
    ],
    "ANTHROPIC_API_KEY": [
        re.compile(r'\bsk-ant-api03-[A-Za-z0-9_\-]{95}\b'),
        re.compile(r'\bsk-ant-[A-Za-z0-9_\-]{90,120}\b'),
    ],
    "HUGGINGFACE_TOKEN": [
        re.compile(r'\bhf_[A-Za-z0-9]{38}\b'),
    ],
    "STRIPE_API_KEY": [
        re.compile(r'\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,}\b'),
    ],
    "SENDGRID_API_KEY": [
        re.compile(r'\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b'),
    ],
    "TWILIO_API_KEY": [
        re.compile(r'\bSK[a-f0-9]{32}\b'),
    ],
    "JWT_TOKEN": [
        re.compile(r'\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b'),
    ],
}

@dataclass
class SecretPattern:
    secret_name: str
    patterns: list[re.Pattern]


class SecretsAnalyzer(BaseDetector):
    """
    Analyzer for detecting secrets in generated text.
    """
    def __init__(self):
        super().__init__()
        self.secrets = self.get_recognizers()

    def get_recognizers(self) -> list[re.Pattern]:
        secrets = []
        for secret_name, regex_pattern in _SECRETS_PATTERNS.items():
            secrets.append(SecretPattern(secret_name, regex_pattern))
        return secrets

    def detect_all(self, text: str) -> list[DetectorResult]:
        res = []
        for secret in self.secrets:
            for pattern in secret.patterns:
                for match in pattern.finditer(text):
                    res.append(DetectorResult(secret.secret_name, match.start(), match.end()))
        return res



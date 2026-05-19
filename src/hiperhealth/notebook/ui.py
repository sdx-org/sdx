"""
title: Public lazy-loading gateway for the optional notebook UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from hiperhealth.notebook._controller import NotebookController

if TYPE_CHECKING:
    from hiperhealth.llm import LLMSettings, StructuredLLM
    from hiperhealth.pipeline import StageRunner

_OPTIONAL_DEPENDENCIES = frozenset({'ipywidgets', 'IPython'})
_INSTALL_MESSAGE = (
    'HiperHealth notebook UI requires optional dependencies. Install them '
    'with: pip install "hiperhealth[notebook]"'
)


class NotebookDependencyError(ImportError):
    """
    title: Raised when optional notebook UI dependencies are unavailable.
    """


class _NotebookWidget(Protocol):
    """
    title: Minimal protocol for a displayable notebook widget.
    """

    def display(self) -> None:
        """
        title: Display the widget in the current notebook.
        """


class _NotebookUIFactory(Protocol):
    """
    title: Constructor protocol for the lazily imported notebook UI.
    """

    def __call__(
        self,
        *,
        controller: NotebookController,
        llm: StructuredLLM | None = None,
        llm_settings: LLMSettings | None = None,
    ) -> _NotebookWidget:
        """
        title: Create a notebook widget instance.
        parameters:
          controller:
            type: NotebookController
          llm:
            type: StructuredLLM | None
          llm_settings:
            type: LLMSettings | None
        returns:
          type: _NotebookWidget
        """


def show(
    data_dir: str | Path | None = None,
    *,
    session_path: str | Path | None = None,
    language: str = 'en',
    runner: StageRunner | None = None,
    llm: StructuredLLM | None = None,
    llm_settings: LLMSettings | None = None,
) -> object:
    """
    title: Display the HiperHealth notebook UI.
    summary: |-
      Optional notebook dependencies are imported lazily so that
      ``from hiperhealth.notebook import ui`` remains safe in a base install.
    parameters:
      data_dir:
        type: str | Path | None
      session_path:
        type: str | Path | None
      language:
        type: str
      runner:
        type: StageRunner | None
      llm:
        type: StructuredLLM | None
      llm_settings:
        type: LLMSettings | None
    returns:
      type: object
    """
    notebook_ui_cls = _load_notebook_ui()
    controller = NotebookController(
        data_dir=data_dir,
        session_path=session_path,
        language=language,
        runner=runner,
    )
    notebook_ui: _NotebookWidget = notebook_ui_cls(
        controller=controller,
        llm=llm,
        llm_settings=llm_settings,
    )
    notebook_ui.display()
    return notebook_ui


def _load_notebook_ui() -> _NotebookUIFactory:
    """
    title: Import the widget implementation with an actionable fallback.
    returns:
      type: _NotebookUIFactory
    """
    try:
        from hiperhealth.notebook._widgets import NotebookUI
    except ModuleNotFoundError as exc:
        if exc.name in _OPTIONAL_DEPENDENCIES:
            raise NotebookDependencyError(_INSTALL_MESSAGE) from exc
        raise
    return NotebookUI


__all__ = ['NotebookDependencyError', 'show']

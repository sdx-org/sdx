import threading

from typing import Any, Callable, Dict, List, TypeVar, cast, overload

from .base import BasePipeline

_REGISTRY: Dict[str, Callable[[], BasePipeline]] = {}


F = TypeVar('F', bound=Callable[[], BasePipeline])


@overload
def register_pipeline(name: str, factory: None = None) -> Callable[[F], F]: ...
@overload
def register_pipeline(name: str, factory: F) -> F: ...


def register_pipeline(
    name: str, factory: Callable[[], BasePipeline] | None = None
) -> Callable[[F], F] | F:
    """Register a pipeline factory under a name.

    Can be used as a decorator::

        @register_pipeline('mock')
        def make_mock():
            return MockPipeline()

    Or called directly::

        register_pipeline('mock', factory)

    The factory should be a zero-arg callable returning a ``BasePipeline``.
    """

    if factory is None:

        def decorator(f: F) -> F:
            _REGISTRY[name] = f
            return f

        return decorator

    _REGISTRY[name] = factory
    return factory


def list_pipelines() -> List[str]:
    """Return a list of registered pipeline names."""
    return list(_REGISTRY.keys())


class LazyPipelineProxy(BasePipeline):
    """Proxy that defers pipeline initialization until first use.

    Thread-safe: initialization uses a lock to ensure the underlying
    pipeline is created and initialized exactly once.
    """

    def __init__(self, factory: Callable[[], BasePipeline]):
        # name reflects factory name if possible
        super().__init__(name=getattr(factory, '__name__', 'lazy_pipeline'))
        self._factory = factory
        self._lock = threading.Lock()
        self._pipeline: BasePipeline | None = None

    def _ensure_init(self) -> None:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    p = self._factory()
                    p.initialize()
                    self._pipeline = p
                    self.initialized = True

    def initialize(self) -> None:
        """Initialize the underlying pipeline instance if not already done."""
        self._ensure_init()

    def process(self, text: str) -> Any:
        """Process `text` using the underlying pipeline.

        Ensures lazy initialization has occurred before delegating to the
        concrete pipeline instance.
        """
        self._ensure_init()
        pipeline = cast(BasePipeline, self._pipeline)
        return pipeline.process(text)

    def shutdown(self) -> None:
        """Shutdown the underlying pipeline if it has been initialized."""
        if self._pipeline is not None:
            self._pipeline.shutdown()

    def health_check(self) -> bool:
        """Return health status of the underlying pipeline.

        This will initialize the pipeline if it has not been initialized yet.
        """
        self._ensure_init()
        pipeline = cast(BasePipeline, self._pipeline)
        return pipeline.health_check()


def get_pipeline(name: str) -> LazyPipelineProxy:
    """Return a lazy proxy for the pipeline registered under `name`.

    Raises ``KeyError`` if no factory is registered under `name`.
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"No pipeline registered under '{name}'")
    return LazyPipelineProxy(factory)

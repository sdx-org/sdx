import threading
from typing import Callable, Dict, List, Any, Optional, Union

from .base import BasePipeline


Factory = Callable[[], BasePipeline]
Decorator = Callable[[Factory], Factory]

_REGISTRY: Dict[str, Factory] = {}


def register_pipeline(name: str, factory: Optional[Factory] = None) -> Union[Decorator, Factory]:
    """Register a pipeline factory under a name.

    Can be used as a decorator:

        @register_pipeline('mock')
        def make_mock():
            return MockPipeline()

    Or called directly:

        register_pipeline('mock', factory)

    The factory should be a zero-arg callable returning a `BasePipeline`.
    """

    if factory is None:
        def decorator(f: Factory) -> Factory:
            _REGISTRY[name] = f
            return f

        return decorator

    _REGISTRY[name] = factory
    return factory


def list_pipelines() -> List[str]:
    return list(_REGISTRY.keys())


class LazyPipelineProxy(BasePipeline):
    """Proxy that defers pipeline initialization until first use.

    Thread-safe: initialization uses a lock to ensure the underlying
    pipeline is created and initialized exactly once.
    """

    def __init__(self, factory: Factory) -> None:
        # name reflects factory name if possible
        super().__init__(name=getattr(factory, "__name__", "lazy_pipeline"))
        self._factory = factory
        self._lock = threading.Lock()
        self._pipeline: Optional[BasePipeline] = None

    def _ensure_init(self) -> None:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    p = self._factory()
                    p.initialize()
                    self._pipeline = p
                    self.initialized = True

    def initialize(self) -> None:
        self._ensure_init()

    def process(self, text: str) -> Any:
        self._ensure_init()
        # _ensure_init guarantees _pipeline is initialized
        assert self._pipeline is not None
        return self._pipeline.process(text)

    def shutdown(self) -> None:
        if self._pipeline is not None:
            self._pipeline.shutdown()

    def health_check(self) -> bool:
        self._ensure_init()
        assert self._pipeline is not None
        return self._pipeline.health_check()


def get_pipeline(name: str) -> BasePipeline:
    factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"No pipeline registered under '{name}'")
    return LazyPipelineProxy(factory)

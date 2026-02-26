"""Simple benchmark to compare eager vs lazy pipeline initialization.

This script uses the mock pipeline to demonstrate timing differences between
creating+initializing a pipeline eagerly vs using the lazy proxy.
"""
import time
from time import perf_counter

from hiperhealth.nlp import get_pipeline, register_pipeline


def bench_lazy(name: str, iters: int = 10):
    p = get_pipeline(name)
    # first call triggers initialization
    t0 = perf_counter()
    for _ in range(iters):
        p.process("hello world")
    t1 = perf_counter()
    return t1 - t0


def bench_eager(factory, iters: int = 10):
    # create and initialize eagerly
    inst = factory()
    inst.initialize()
    t0 = perf_counter()
    for _ in range(iters):
        inst.process("hello world")
    t1 = perf_counter()
    return t1 - t0


def main():
    # Use the built-in mock pipeline (registered as "mock")
    print("Running nlp bench (mock pipeline)")
    lazy_time = bench_lazy("mock", iters=1000)
    print(f"Lazy total time (1000 iters): {lazy_time:.4f}s")

    # find factory from registry via import (internal; for demo only)
    from hiperhealth.nlp import registry

    factory = registry._REGISTRY.get("mock")
    if factory is None:
        print("mock factory not found; ensure mock pipeline is registered")
        return
    eager_time = bench_eager(factory, iters=1000)
    print(f"Eager total time (1000 iters): {eager_time:.4f}s")


if __name__ == "__main__":
    main()

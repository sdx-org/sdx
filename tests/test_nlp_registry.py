import threading


def test_register_and_lazy_init(tmp_path, monkeypatch):
    from hiperhealth.nlp import registry

    calls = {"init": 0}

    def factory():
        class _P:
            def __init__(self):
                self.initialized = False

            def initialize(self):
                calls["init"] += 1
                self.initialized = True

            def process(self, text: str):
                return text.upper()

            def shutdown(self):
                self.initialized = False

            def health_check(self):
                return True

        return _P()

    registry.register_pipeline("test_lazy", factory)

    p = registry.get_pipeline("test_lazy")
    # not initialized before use
    assert not p.initialized

    out = p.process("hello")
    assert out == "HELLO"
    assert p.initialized
    # factory initialize called exactly once
    assert calls["init"] == 1


def test_threaded_init(monkeypatch):
    from hiperhealth.nlp import registry

    initialized = {"count": 0}

    def factory():
        class P:
            def __init__(self):
                self.initialized = False

            def initialize(self):
                # simulate some work
                import time

                time.sleep(0.01)
                initialized["count"] += 1
                self.initialized = True

            def process(self, text: str):
                return text

            def shutdown(self):
                self.initialized = False

            def health_check(self):
                return True

        return P()

    registry.register_pipeline("thread_lazy", factory)
    proxy = registry.get_pipeline("thread_lazy")

    results = []

    def worker():
        results.append(proxy.process("x"))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # initialization should happen exactly once
    assert initialized["count"] == 1

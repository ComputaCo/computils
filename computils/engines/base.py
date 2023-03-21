import threading


class CompletionEngine:

    _lock: threading.Lock

    def __init__(self, no_multithreading=False):
        self._lock = threading.Lock() if no_multithreading else None

    def complete(self, text, *args, **kwargs) -> str:
        if self._lock:
            with self._lock:
                self.complete(text, *args, **kwargs)
        else:
            self.complete(text, *args, **kwargs)

    def _complete(self, text, *args, **kwargs) -> str:
        raise NotImplementedError()


class ConversationEngine:

    _lock: threading.Lock

    def __init__(self, no_multithreading=False):
        self._lock = threading.Lock() if no_multithreading else None

    def chat(self, messages, *args, **kwargs) -> str:
        if self._lock:
            with self._lock:
                self.chat(messages, *args, **kwargs)
        else:
            self.chat(messages, *args, **kwargs)

    def _chat(self, messages, *args, **kwargs) -> str:
        raise NotImplementedError()

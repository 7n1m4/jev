"""A pool of scorer subprocesses, so concurrent requests do not serialise.

One DecisionRuntime owns exactly one scorer subprocess and must never be used
from two threads at once: its stdin is a single JSON-lines stream, and two
interleaved writers would corrupt the protocol and pair the wrong response with
the wrong request -- which would return a confidently wrong answer, the worst
possible failure for a decision service.

So requests pick one runtime out of the pool; a request never shares a runtime.
Each worker is guarded by its own lock as a belt-and-braces check that a single
worker is only ever decoding one request at a time.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Optional

from jevreadout import Decision, DecisionRuntime


class Worker:
    """One scorer subprocess, with a lock and liveness tracking."""

    def __init__(self, slots: int = 16, **kw: Any) -> None:
        self.lock = threading.Lock()
        self.slots = slots
        self._kw = kw
        self.rt: Optional[DecisionRuntime] = None
        self.loaded_at: Optional[float] = None
        self.requests = 0
        self._start()

    def _start(self) -> None:
        self.rt = DecisionRuntime(**self._kw)
        self.loaded_at = time.time()

    def alive(self) -> bool:
        if self.rt is None or self.rt._proc is None:
            return False
        return self.rt._proc.poll() is None

    def restart(self) -> None:
        try:
            if self.rt is not None:
                self.rt.close()
        except Exception:
            pass
        self._start()

    def decide(self, *args: Any, **kw: Any) -> Decision:
        if not self.alive():
            raise RuntimeError("scorer worker died")
        with self.lock:
            if not self.alive():
                raise RuntimeError("scorer worker died")
            d = self.rt.decide(*args, **kw)  # type: ignore[union-attr]
            self.requests += 1
            return d


class ScorerPool:
    """Round-robins requests over N scorer subprocesses.

    A free queue is used rather than an index: taking an element out of a queue
    is atomic in CPython, so no interleaving is possible even if two threads
    race, and an idle worker is preferred over creating load imbalance.
    """

    def __init__(self, n: int = 2, slots: int = 16, **kw: Any) -> None:
        self.n = n
        self._free: "queue.LifoQueue[Worker]" = queue.LifoQueue()
        self.workers = [Worker(slots=slots, **kw) for _ in range(n)]
        for w in self.workers:
            self._free.put(w)
        self._restarts = 0
        self._restarts_lock = threading.Lock()

    def acquire(self) -> Worker:
        w = self._free.get()
        return w

    def release(self, w: Worker) -> None:
        self._free.put(w)

    def decide(self, *args: Any, **kw: Any) -> tuple:
        """Returns (decision, worker). The caller MUST release the worker."""
        w = self.acquire()
        try:
            return w.decide(*args, **kw), w
        except Exception:
            # a dead scorer is an internal failure (500), never a wrong answer:
            # restart the worker so the next request does not fail too.
            with self._restarts_lock:
                self._restarts += 1
            try:
                w.restart()
            except Exception:
                pass
            raise
        finally:
            self.release(w)

    def close(self) -> None:
        for w in self.workers:
            try:
                w.rt.close()
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "workers": self.n,
            "alive": sum(1 for w in self.workers if w.alive()),
            "restarts": self._restarts,
            "per_worker_requests": [w.requests for w in self.workers],
        }


__all__ = ["ScorerPool", "Worker"]

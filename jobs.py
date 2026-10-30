"""Bounded in-process jobs; workers never access Streamlit session state."""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Callable, TypeVar

T = TypeVar("T")
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="specta")
_capacity = BoundedSemaphore(8)


def submit(function: Callable[[], T]) -> Future[T]:
    if not _capacity.acquire(blocking=False):
        raise RuntimeError("AI work queue is full. Observations continue; try again after pending work finishes.")
    try:
        future = _pool.submit(function)
    except RuntimeError:
        _capacity.release()
        raise
    future.add_done_callback(lambda _: _capacity.release())
    return future

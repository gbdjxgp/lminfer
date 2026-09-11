from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from lminfer.v1.request import Request


class FCFSRequestQueue:
    def __init__(self) -> None:
        self._queue: deque[Request] = deque()

    def add_request(self, request: Request) -> None:
        self._queue.append(request)

    def prepend_request(self, request: Request) -> None:
        self._queue.appendleft(request)

    def pop_request(self) -> Request:
        return self._queue.popleft()

    def peek_request(self) -> Request:
        return self._queue[0]

    def remove_request(self, request: Request) -> None:
        self._queue.remove(request)

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._queue)

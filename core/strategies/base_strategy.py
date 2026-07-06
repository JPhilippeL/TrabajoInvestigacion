from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]


class BaseStrategy(ABC):
    def __init__(self, facade):
        self.facade = facade

    @abstractmethod
    def execute(
        self,
        config: Any,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        pass

from typing import Any

from core.strategies.base_strategy import BaseStrategy, LogCallback, ProgressCallback


class PredictStrategy(BaseStrategy):
    def execute(
        self,
        config: Any,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> Any:
        self.facade.predict(config, log_callback, progress_callback)

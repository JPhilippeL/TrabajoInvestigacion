from PySide6.QtCore import QThread, Signal

from CheapNet_GUI.model.graph_generation import generate_all_graphs


class _ThreadLoggerProxy:
    def __init__(self, emit_func):
        self._emit = emit_func

    def info(self, *args):
        if not args:
            return
        message = " ".join(str(arg) for arg in args)
        self._emit(message)

    def error(self, *args):
        if not args:
            return
        message = " ".join(str(arg) for arg in args)
        self._emit(f"ERROR: {message}")


class DBGenerationThread(QThread):
    finished_success = Signal()
    finished_error = Signal(str)
    log_message = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            thread_logger = _ThreadLoggerProxy(self.log_message.emit)
            generate_all_graphs(**self.params, log_callback=thread_logger)
            self.finished_success.emit()
        except Exception as e:
            self.finished_error.emit(str(e))

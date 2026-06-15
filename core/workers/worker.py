from PySide6.QtCore import QThread, Signal, Slot


class Worker(QThread):
    result_ready = Signal(object)
    log = Signal(str)
    progress = Signal(int)
    error = Signal(str)

    def __init__(self, config, strategy, parent=None):
        super(Worker, self).__init__(parent)
        self.config = config
        self.strategy = strategy

    @Slot()
    def run(self):
        try:
            result = self.strategy.execute(self.config, self.log.emit, self.progress.emit)
            self.result_ready.emit(result)
        except Exception as e:
            self.error.emit(str(e))

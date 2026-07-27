"""Reusable background worker threads for GUI widgets.

Any network/data-fetch call that would otherwise block the Qt event loop
should be routed through FetchWorker rather than called directly from a
slot connected to a UI signal.
"""
from PyQt6.QtCore import QThread, pyqtSignal
from ..utils.logger import logger


class FetchWorker(QThread):
    """Runs an arbitrary callable off the GUI thread and reports the result."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Background fetch error: {str(e)}")
            self.error.emit(str(e))

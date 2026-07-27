"""Reusable background workers for GUI widgets.

Any network/data-fetch call that would otherwise block the Qt event loop
should be routed through FetchWorker rather than called directly from a
slot connected to a UI signal.

FetchWorker deliberately does NOT subclass QThread: yfinance's HTTP
backend (curl_cffi) reproducibly segfaults (0xc0000005 in python310.dll)
when the call is made inside a QThread.run() and the result is then
emitted as that thread winds down - Qt's ownership/teardown of the OS
thread races with curl_cffi's own thread-local cleanup. A plain
threading.Thread has no such conflict since Qt never takes ownership of
it; PyQt's signal/slot queued-connection mechanism works correctly from
any Python thread, not just QThread, so this keeps the same signal-based
interface call sites already use.
"""
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from ..utils.logger import logger


class FetchWorker(QObject):
    """Runs an arbitrary callable on a plain background thread and reports the result via Qt signals."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def isRunning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Background fetch error: {str(e)}")
            self.error.emit(str(e))

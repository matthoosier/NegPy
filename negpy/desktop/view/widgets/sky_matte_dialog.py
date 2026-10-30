"""Transient SegFormer sky-matte preview."""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import hint_label, pin_dialog_default
from negpy.desktop.view.widgets.update_dialog import _own
from negpy.kernel.system.i18n import tr
from negpy.kernel.system.logging import get_logger
from negpy.services.assets.sky_matte import SkyMatteError, SkyMatteModel, make_sky_matte

logger = get_logger(__name__)


class SkyMatteWorker(QThread):
    progress = pyqtSignal(int, int)
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image: np.ndarray) -> None:
        super().__init__()
        self._image = image
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            matte = make_sky_matte(
                self._image,
                SkyMatteModel(),
                self._cancelled.is_set,
                self.progress.emit,
            )
        except SkyMatteError as exc:
            if not self._cancelled.is_set():
                logger.error("Sky matte preview failed: %s", exc)
                self.failed.emit(tr("Sky matte failed. See the log for details."))
        except Exception:
            logger.exception("Sky matte preview failed")
            if not self._cancelled.is_set():
                self.failed.emit(tr("Sky matte failed. See the log for details."))
        else:
            if not self._cancelled.is_set():
                self.ready.emit(matte)


class SkyMatteDialog(QDialog):
    """Modeless matte preview; its pixels are never stored in the edit."""

    def __init__(
        self,
        image: Optional[np.ndarray],
        is_current: Callable[[], bool],
        parent: Optional[QWidget] = None,
        error: str = "",
    ) -> None:
        super().__init__(parent)
        self._is_current = is_current
        self._worker: Optional[SkyMatteWorker] = None
        self.setWindowTitle(tr("Sky Matte"))
        self.resize(860, 640)

        layout = QVBoxLayout(self)
        self.status = hint_label("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.addStretch()
        close_button = QPushButton(tr("Close"))
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        pin_dialog_default(None, close_button, scope=self)
        layout.addLayout(footer)

        if error:
            self._set_error(error)
        elif image is None:
            self._set_error(tr("Open a rendered frame before showing the sky matte."))
        else:
            self._start(image)

    def _start(self, image: np.ndarray) -> None:
        self.status.setText(tr("Finding the sky…"))
        self.progress.setVisible(True)
        worker = SkyMatteWorker(image)
        self._worker = worker
        worker.progress.connect(self._on_progress)
        worker.ready.connect(self._on_ready)
        worker.failed.connect(self._set_error)
        _own(worker)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(lambda: worker.wait())
        worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self.status.setText(tr("Downloading the sky model…"))
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(done)

    def _on_ready(self, matte: np.ndarray) -> None:
        if not self._is_current():
            self._set_error(tr("The frame changed. Close this window and show the sky matte again."))
            return
        gray = np.clip(matte * 255.0 + 0.5, 0, 255).astype(np.uint8)
        h, w = gray.shape
        image = QImage(gray.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.image_label.adjustSize()
        self.progress.setVisible(False)
        self.status.setText(tr("White shows sky coverage. Black shows no coverage."))

    def _set_error(self, message: str) -> None:
        self.progress.setVisible(False)
        if self.status.property("hint") != "error":
            error_label = hint_label(kind="error")
            error_label.setWordWrap(True)
            layout = self.layout()
            assert layout is not None
            layout.replaceWidget(self.status, error_label)
            self.status.deleteLater()
            self.status = error_label
        self.status.setText(message)

    def invalidate(self) -> None:
        if self._is_current():
            return
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        self._set_error(tr("The frame changed. Close this window and show the sky matte again."))

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        super().reject()

    def closeEvent(self, a0: QCloseEvent | None) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        super().closeEvent(a0)

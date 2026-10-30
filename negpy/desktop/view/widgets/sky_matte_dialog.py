"""Transient SegFormer sky-matte preview."""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import hint_label, pin_dialog_default
from negpy.desktop.view.widgets.update_dialog import _own
from negpy.kernel.system.i18n import tr
from negpy.kernel.system.logging import get_logger
from negpy.services.assets.sky_matte import SkyMatteError, SkyMatteModel, make_sky_matte_comparison

logger = get_logger(__name__)


class _MatteImageLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._source: Optional[QImage] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(1, 1)

    def set_image(self, image: QImage) -> None:
        self._source = image.copy()
        self._refresh()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._source is not None and not self.size().isEmpty():
            self.setPixmap(
                QPixmap.fromImage(self._source).scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class SkyMatteWorker(QThread):
    progress = pyqtSignal(int, int)
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, final_positive: np.ndarray, normalized_log: np.ndarray) -> None:
        super().__init__()
        self._final_positive = final_positive
        self._normalized_log = normalized_log
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            result = make_sky_matte_comparison(
                self._final_positive,
                self._normalized_log,
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
                self.ready.emit(result)


class SkyMatteDialog(QDialog):
    """Modeless matte preview; its pixels are never stored in the edit."""

    def __init__(
        self,
        final_positive: Optional[np.ndarray],
        normalized_log: Optional[np.ndarray],
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

        self.images_widget = QWidget()
        self.images_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, self.images_widget)
        self.images_layout.setContentsMargins(0, 0, 0, 0)
        self.images_layout.setSpacing(8)
        self.image_labels: list[_MatteImageLabel] = []
        self.detail_labels: list[QLabel] = []
        for index, title in enumerate((tr("Final Positive"), tr("Rough Positive"), tr("Matte Comparison"))):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_title = QLabel(title)
            panel_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            panel_layout.addWidget(panel_title)
            details = QLabel()
            details.setAlignment(Qt.AlignmentFlag.AlignCenter)
            details.setWordWrap(True)
            panel_layout.addWidget(details)
            self.detail_labels.append(details)
            image_label = _MatteImageLabel()
            panel_layout.addWidget(image_label, 1)
            self.images_layout.addWidget(panel, 1)
            self.image_labels.append(image_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.images_widget)
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
        elif final_positive is None or normalized_log is None:
            self._set_error(tr("Open a rendered frame before showing the sky matte."))
        else:
            self._start(final_positive, normalized_log)

    def _start(self, final_positive: np.ndarray, normalized_log: np.ndarray) -> None:
        self._final_positive_shape = final_positive.shape[:2]
        self._normalized_log_shape = normalized_log.shape[:2]
        self.status.setText(tr("Comparing sky mattes…"))
        self.progress.setVisible(True)
        worker = SkyMatteWorker(final_positive, normalized_log)
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

    def _on_ready(self, result: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        if not self._is_current():
            self._set_error(tr("The frame changed. Close this window and show the sky matte again."))
            return
        final_matte, rough_matte, comparison = result
        height, width = final_matte.shape
        self._panel_orientation = Qt.Orientation.Horizontal if height > width else Qt.Orientation.Vertical
        self.images_layout.setDirection(
            QBoxLayout.Direction.LeftToRight if self._panel_orientation == Qt.Orientation.Horizontal else QBoxLayout.Direction.TopToBottom
        )
        final_gray = np.clip(final_matte * 255.0 + 0.5, 0, 255).astype(np.uint8)
        rough_gray = np.clip(rough_matte * 255.0 + 0.5, 0, 255).astype(np.uint8)
        comparison_rgb = np.clip(comparison * 255.0 + 0.5, 0, 255).astype(np.uint8)
        final_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        final_rgb[:, :, 1] = final_gray
        rough_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rough_rgb[:, :, 0] = rough_gray
        self.image_labels[0].set_image(QImage(final_rgb.data, width, height, width * 3, QImage.Format.Format_RGB888))
        self.image_labels[1].set_image(QImage(rough_rgb.data, width, height, width * 3, QImage.Format.Format_RGB888))
        self.image_labels[2].set_image(QImage(comparison_rgb.data, width, height, width * 3, QImage.Format.Format_RGB888))
        final_height, final_width = self._final_positive_shape
        rough_height, rough_width = self._normalized_log_shape
        self.detail_labels[0].setText(
            "\n".join(
                (
                    tr("Input image: {width} × {height}", width=final_width, height=final_height),
                    tr("Matte: {width} × {height}", width=width, height=height),
                )
            )
        )
        self.detail_labels[1].setText(
            "\n".join(
                (
                    tr("Input image: {width} × {height}", width=rough_width, height=rough_height),
                    tr("Matte: {width} × {height}", width=width, height=height),
                )
            )
        )
        self.detail_labels[2].setText(
                tr("Matte: {width} x {height}", width=comparison.shape[1], height=comparison.shape[0])
        )
        self.progress.setVisible(False)
        self.status.setText(tr("Green shows final-positive coverage. Red shows rough-positive coverage. Yellow shows overlap."))

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

import os
import sys
import json
import time

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_XCB_GL_INTEGRATION"] = "none"
os.environ["QT_OPENGL"] = "software"

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QTextEdit, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QTextCursor
from PyQt6.QtCore import Qt, QTimer


OUTPUT_DIR = "./sam3_live_output"
PRED_PATH = os.path.join(OUTPUT_DIR, "latest_pred.jpg")
DEPTH_PATH = os.path.join(OUTPUT_DIR, "latest_depth.jpg")
STATE_PATH = os.path.join(OUTPUT_DIR, "state.json")
LOG_PATH = os.path.join(OUTPUT_DIR, "viewer.log")


class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Viewer (Separated Process)")

        self.left_title = QLabel("SAM3 Prediction")
        self.right_title = QLabel("Depth + SAM3 Mask")

        for title in [self.left_title, self.right_title]:
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setMaximumHeight(36)
            title.setStyleSheet("""
                background-color: #f0f0f0;
                color: black;
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #cccccc;
            """)

        self.left_label = QLabel("Waiting for pred image...")
        self.right_label = QLabel("Waiting for depth image...")
        self.status = QLabel("Ready")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.left_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.right_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for lbl in [self.left_label, self.right_label]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #111111; color: white;")

        self.status.setMaximumHeight(32)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("font-size: 15px; padding: 4px;")

        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.clicked.connect(self.refresh_all)

        self.log_box.setMaximumHeight(220)
        self.log_box.setStyleSheet("background-color: #0b0b0b; color: #dddddd;")

        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(0, 0, 0, 0)
        left_panel.setSpacing(0)
        left_panel.addWidget(self.left_title, 0)
        left_panel.addWidget(self.left_label, 1)

        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 0, 0, 0)
        right_panel.setSpacing(0)
        right_panel.addWidget(self.right_title, 0)
        right_panel.addWidget(self.right_label, 1)

        layout_img = QHBoxLayout()
        layout_img.setContentsMargins(0, 0, 0, 0)
        layout_img.setSpacing(6)
        layout_img.addLayout(left_panel, 1)
        layout_img.addLayout(right_panel, 1)

        layout = QVBoxLayout()
        layout.addLayout(layout_img, 4)
        layout.addWidget(self.status, 0)
        layout.addWidget(self.refresh_btn, 0)
        layout.addWidget(self.log_box, 1)

        self.setLayout(layout)

        self.current_pred_pixmap = None
        self.current_depth_pixmap = None
        self.last_log_size = 0
        self.last_state_mtime = 0
        self.last_pred_mtime = 0
        self.last_depth_mtime = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_all)
        self.timer.start(200)

    def set_pixmap_fit(self, label, pixmap):
        if pixmap is None or pixmap.isNull():
            return
        scaled = pixmap.scaled(
            max(1, label.width()),
            max(1, label.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)

    def load_image_if_updated(self, path, last_mtime_attr, label_name):
        if not os.path.exists(path):
            return

        mtime = os.path.getmtime(path)
        if getattr(self, last_mtime_attr) == mtime:
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            return

        setattr(self, last_mtime_attr, mtime)

        if label_name == "pred":
            self.current_pred_pixmap = pixmap
            self.set_pixmap_fit(self.left_label, pixmap)
        else:
            self.current_depth_pixmap = pixmap
            self.set_pixmap_fit(self.right_label, pixmap)

    def load_state_if_updated(self):
        if not os.path.exists(STATE_PATH):
            return

        mtime = os.path.getmtime(STATE_PATH)
        if self.last_state_mtime == mtime:
            return

        self.last_state_mtime = mtime

        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            return

        if state.get("status") == "running":
            self.status.setText(
                f"Running | pred_fps={state.get('pred_fps', 0)} | "
                f"capture_fps={state.get('capture_fps', 0)} | "
                f"set_image={state.get('set_image_ms', 0)}ms | "
                f"prompt={state.get('prompt_ms', 0)}ms | "
                f"objects={state.get('objects', 0)}"
            )
        elif state.get("status") == "error":
            self.status.setText(f"Error | {state.get('error', '')}")
        else:
            self.status.setText(state.get("status", "Unknown"))

    def load_log_incremental(self):
        if not os.path.exists(LOG_PATH):
            return

        try:
            size = os.path.getsize(LOG_PATH)
            if size < self.last_log_size:
                self.last_log_size = 0

            with open(LOG_PATH, "r", encoding="utf-8") as f:
                f.seek(self.last_log_size)
                new_text = f.read()
                self.last_log_size = f.tell()

            if new_text:
                self.log_box.moveCursor(QTextCursor.MoveOperation.End)
                self.log_box.insertPlainText(new_text)
                self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        except Exception:
            pass

    def refresh_all(self):
        self.load_image_if_updated(PRED_PATH, "last_pred_mtime", "pred")
        self.load_image_if_updated(DEPTH_PATH, "last_depth_mtime", "depth")
        self.load_state_if_updated()
        self.load_log_incremental()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pred_pixmap is not None:
            self.set_pixmap_fit(self.left_label, self.current_pred_pixmap)
        if self.current_depth_pixmap is not None:
            self.set_pixmap_fit(self.right_label, self.current_depth_pixmap)


def main():
    app = QApplication(sys.argv)
    viewer = Viewer()
    viewer.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
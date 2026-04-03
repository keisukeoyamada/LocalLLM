# chat_gui.py
# PyQt6-based chat GUI with pixel art character and MLX LLM backend.

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from mlx_lm import generate, load
from mlx_lm.generate import make_sampler

# --- Config ---
MLX_MODEL_PATH = "./models/gemma-3-27b-mlx"
WAITING_IMG = "./character_image/waiting.png"
GENERATE_IMG = "./character_image/generate.png"
CHARACTER_WIDTH = 400  # display width (height auto-scaled)

SYSTEM_PROMPT = "あなたは日本語で回答するアシスタントです。常に日本語で答えてください。"


class GenerateThread(QThread):
    """Run LLM generation in background to keep UI responsive."""
    finished = pyqtSignal(str)

    def __init__(self, model, tokenizer, prompt):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt

    def run(self):
        response = generate(
            self.model,
            self.tokenizer,
            prompt=self.prompt,
            max_tokens=2000,
            sampler=make_sampler(temp=0.7),
            verbose=True,
        )
        self.finished.emit(response)


class MainWindow(QMainWindow):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.thread = None

        self.setWindowTitle("Local LLM Chat")
        self.setMinimumWidth(600)

        # --- Pixmaps ---
        self.pixmap_waiting = QPixmap(WAITING_IMG).scaledToWidth(
            CHARACTER_WIDTH, Qt.TransformationMode.SmoothTransformation
        )
        self.pixmap_generate = QPixmap(GENERATE_IMG).scaledToWidth(
            CHARACTER_WIDTH, Qt.TransformationMode.SmoothTransformation
        )

        # --- Layout ---
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Character image
        self.char_label = QLabel()
        self.char_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.char_label.setPixmap(self.pixmap_waiting)
        layout.addWidget(self.char_label)

        # Chat history
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(200)
        layout.addWidget(self.chat_display)

        # Input row
        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("メッセージを入力...")
        self.input_field.returnPressed.connect(self.send_message)
        input_row.addWidget(self.input_field)

        self.send_btn = QPushButton("送信")
        self.send_btn.clicked.connect(self.send_message)
        input_row.addWidget(self.send_btn)

        layout.addLayout(input_row)

    def send_message(self):
        user_input = self.input_field.text().strip()
        if not user_input:
            return

        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)

        # Show user message
        self.chat_display.append(f"<b>You:</b> {user_input}")

        # Switch to generate image
        self.char_label.setPixmap(self.pixmap_generate)

        # Build prompt and run in background thread
        full_prompt = f"{SYSTEM_PROMPT}\n\nユーザー: {user_input}\nアシスタント:"
        self.thread = GenerateThread(self.model, self.tokenizer, full_prompt)
        self.thread.finished.connect(self.on_response)
        self.thread.start()

    def on_response(self, response):
        self.chat_display.append(f"<b>AI:</b> {response}\n")
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

        # Switch back to waiting image
        self.char_label.setPixmap(self.pixmap_waiting)

        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()


if __name__ == "__main__":
    print(f"Loading model: {MLX_MODEL_PATH}")
    model, tokenizer = load(MLX_MODEL_PATH)

    app = QApplication(sys.argv)
    window = MainWindow(model, tokenizer)
    window.show()
    sys.exit(app.exec())

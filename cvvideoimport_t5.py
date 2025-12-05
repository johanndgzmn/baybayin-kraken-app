from PyQt6.uic import loadUi
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

import cv2
import subprocess
from PIL import Image
from kraken import binarization
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


# ----------------------------------------------------
# Load T5 model ONCE (8-bit, auto device)
# ----------------------------------------------------
print("Loading T5 post-processing model...")

t5_tokenizer = T5Tokenizer.from_pretrained("./t5_filspell_cpu_fp32")

t5_model = T5ForConditionalGeneration.from_pretrained(
    "./t5_filspell_cpu_fp32",
    device_map="auto",
    load_in_8bit=True
)

def t5_correct(line: str) -> str:
    """Apply T5 correction on one text line."""
    if not line.strip():
        return ""

    prompt = "fix: " + line
    inputs = t5_tokenizer(prompt, return_tensors="pt").to(t5_model.device)

    outputs = t5_model.generate(
        inputs["input_ids"],
        max_length=64,
        num_beams=4,
        early_stopping=True
    )

    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ----------------------------------------------------
# Main PyQt Application
# ----------------------------------------------------
class Camera:
    def __init__(self):
        self.mainui = loadUi('baybayin-kraken-app_post_process.ui')
        self.mainui.show()
        
        self.mainui.startStopCameraButton.clicked.connect(self.closeEvent)
        self.mainui.captureImageButton.clicked.connect(self.capture_image)
        self.mainui.transliterateImageButton.clicked.connect(self.transliterate_image)
        self.mainui.binarizeImageButton.clicked.connect(self.binarize_image)
        self.mainui.resetCameraButton.clicked.connect(self.reset_camera)
        self.mainui.showOutputTextButton.clicked.connect(self.show_output_text)
        self.mainui.postProcessButton.clicked.connect(self.post_process_text)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print("Error: Cannot access the webcam.")
            return

        self.video_label = self.mainui.cameraLabel
        self.video_label.setScaledContents(True)
        self.video_label.setFixedSize(429, 329)
        self.mainui.cameraLayout.addWidget(self.video_label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_frame = None
        self.frozen = False


    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()

            # Convert for Qt display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))


    def capture_image(self):
        if self.current_frame is not None:
            cv2.imwrite("test_image.png", self.current_frame)
            self.mainui.ocrOutputBox.setPlainText("Image captured!")
            self.timer.stop()
            self.frozen = True


    def binarize_image(self):
        image = Image.open("test_image.png")
        bin_img = binarization.nlbin(image, low=5, high=25)
        bin_img.save("test_binarized.png")
        self.mainui.ocrOutputBox.setPlainText("Image binarized!")


    def transliterate_image(self):
        import os
        os.environ["PYTHONUTF8"] = "1"  # force utf-8 mode for subprocess

        command = [
            "kraken", "-i", "test_binarized.png", "test.txt", "segment", "ocr",
            "-m", "baybayin_custom_dataset.mlmodel_best.mlmodel",
        ]

        log_file = "transliteration_log.txt"

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",   # decode output as UTF-8
                errors="replace"    # replace bad chars instead of crashing
            )

            # Write all logs to file
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("STDOUT:\n" + (result.stdout or "") + "\n")
                f.write("STDERR:\n" + (result.stderr or "") + "\n")

            self.mainui.ocrOutputBox.setPlainText("Transliteration finished. Logs saved to transliteration_log.txt.")
        
        except subprocess.CalledProcessError as e:
            # Write error logs to file as well
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("STDOUT:\n" + (e.stdout or "") + "\n")
                f.write("STDERR:\n" + (e.stderr or "") + "\n")

            self.mainui.ocrOutputBox.setPlainText("Error during OCR. See transliteration_log.txt for details.")


    def post_process_text(self):
        """Runs T5 correction on test.txt and outputs to outputText box."""
        try:
            with open("test.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()

            corrected = []
            for line in lines:
                fixed = t5_correct(line.strip())
                corrected.append(fixed)

            # Save corrected file
            with open("corrected_output.txt", "w", encoding="utf-8") as f:
                for text in corrected:
                    f.write(text + "\n")
            t5_log_file = "t5_logs.txt"

            # Display corrected text in GUI
            self.mainui.outputText.setPlainText("\n".join(corrected))
            self.mainui.ocrOutputBox.setPlainText("Post-Processing complete!")

        except FileNotFoundError:
            self.mainui.ocrOutputBox.setPlainText("Error: test.txt not found.")

        except Exception as e:
            self.mainui.ocrOutputBox.setPlainText(f"Post-processing error:\n{str(e)}")


    def reset_camera(self):
        if self.frozen:
            self.timer.start(30)
            self.frozen = False
            self.mainui.ocrOutputBox.setPlainText("Camera reset. Preview resumed.")


    def show_output_text(self):
        try:
            with open("test.txt", "r", encoding="utf-8") as f:
                self.mainui.outputText.setPlainText(f.read())
        except:
            self.mainui.outputText.setPlainText("Error: test.txt not found.")


    def closeEvent(self):
        self.cap.release()
        self.mainui.close()


if __name__ == "__main__":
    app = QApplication([])
    main = Camera()
    app.exec()

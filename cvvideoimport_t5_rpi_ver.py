from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from picamera2 import Picamera2
import cv2
import subprocess
from PIL import Image
from kraken import binarization
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

os.environ["G_MESSAGES_DEBUG"] = "none"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/arm-linux-gnueabihf/qt5/plugins/platforms"


# ==========================================================
#                 T5 POST-PROCESSING LOAD
# ==========================================================
print("Loading T5 CPU model (RPi-safe)...")

t5_tokenizer = T5Tokenizer.from_pretrained("./t5_filspell_cpu_fp32")

t5_model = T5ForConditionalGeneration.from_pretrained(
    "./t5_filspell_cpu_fp32",
    torch_dtype=torch.float32  # Force CPU
)
t5_model.eval()

def t5_correct(line: str) -> str:
    if not line.strip():
        return ""

    prompt = "fix: " + line
    inputs = t5_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():  # important for RPi speed
        outputs = t5_model.generate(
            inputs["input_ids"],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ==========================================================
#                       MAIN CAMERA APP
# ==========================================================
class Camera:
    def __init__(self):
        self.mainui = loadUi('baybayin-kraken-app_postprocess_rpi.ui')
        self.mainui.show()
        
        # Camera
        self.picam2 = Picamera2()
        self.picam2.configure(
            self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (640, 480)}
            )
        )
        self.picam2.start()

        # Buttons
        self.mainui.startStopCameraButton.clicked.connect(self.closeEvent)
        self.mainui.captureImageButton.clicked.connect(self.capture_image)
        self.mainui.binarizeImageButton.clicked.connect(self.binarize_image)
        self.mainui.transliterateImageButton.clicked.connect(self.transliterate_image)
        self.mainui.resetCameraButton.clicked.connect(self.reset_camera)

        # --- NEW ---
        self.mainui.postProcessButton.clicked.connect(self.post_process_text)
        self.mainui.showOutputTextButton.clicked.connect(self.show_output_text)

        # Display
        self.video_label = self.mainui.cameraLabel
        self.video_label.setScaledContents(True)
        self.video_label.setFixedSize(429, 329)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_frame = None
        self.frozen = False


    # ------------------------------------------------------
    def update_frame(self):
        frame = self.picam2.capture_array()
        self.current_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))


    # ------------------------------------------------------
    def capture_image(self):
        if self.current_frame is not None:
            cv2.imwrite("test_image.png", self.current_frame)
            self.mainui.ocrOutputBox.setPlainText("Image captured!")
            self.timer.stop()
            self.frozen = True


    # ------------------------------------------------------
    def binarize_image(self):
        image = Image.open("test_image.png")
        bin_img = binarization.nlbin(image, low=5, high=25)
        bin_img.save("test_binarized.png")
        self.mainui.ocrOutputBox.setPlainText("Image binarized!")


    # ------------------------------------------------------
    def transliterate_image(self):
        os.environ["PYTHONUTF8"] = "1"

        command = [
            "kraken", "-i", "test_binarized.png", "test.txt",
            "segment", "ocr",
            "-m", "baybayin_custom_dataset_edited.mlmodel_best.mlmodel"
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True
            )

            with open("transliteration_log.txt", "w", encoding="utf-8") as f:
                f.write("STDOUT:\n" + (result.stdout or "") + "\n")
                f.write("STDERR:\n" + (result.stderr or "") + "\n")

            self.mainui.ocrOutputBox.setPlainText("Transliteration finished.")

        except subprocess.CalledProcessError as e:
            with open("transliteration_log.txt", "w", encoding="utf-8") as f:
                f.write("STDOUT:\n" + (e.stdout or "") + "\n")
                f.write("STDERR:\n" + (e.stderr or "") + "\n")
            self.mainui.ocrOutputBox.setPlainText("OCR Error. See logs.")


    # ------------------------------------------------------
    def post_process_text(self):
        """Runs T5 post-processing and writes corrected_output.txt."""
        try:
            with open("test.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()

            corrected = []
            for line in lines:
                fixed = t5_correct(line.strip())
                corrected.append(fixed)

            with open("corrected_output.txt", "w", encoding="utf-8") as f:
                for t in corrected:
                    f.write(t + "\n")

            self.mainui.outputText.setPlainText("\n".join(corrected))
            self.mainui.ocrOutputBox.setPlainText("Post-processing complete!")

        except Exception as e:
            self.mainui.ocrOutputBox.setPlainText(f"Error in post-processing:\n{str(e)}")


    # ------------------------------------------------------
    def show_output_text(self):
        try:
            with open("test.txt", "r", encoding="utf-8") as f:
                self.mainui.outputText.setPlainText(f.read())
        except:
            self.mainui.outputText.setPlainText("test.txt not found.")


    # ------------------------------------------------------
    def reset_camera(self):
        if self.frozen:
            self.timer.start(30)
            self.frozen = False
            self.mainui.ocrOutputBox.setPlainText("Camera resumed.")


    # ------------------------------------------------------
    def closeEvent(self):
        self.picam2.stop()


if __name__ == '__main__':
    app = QApplication([])
    main = Camera()
    app.exec()

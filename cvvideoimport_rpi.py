from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from picamera2 import Picamera2
import cv2
import time
import subprocess
from PIL import Image
from kraken import binarization
import os
import atexit
import board
import neopixel

os.environ["G_MESSAGES_DEBUG"] = "none"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/arm-linux-gnueabihf/qt5/plugins/platforms"

pixels = neopixel.NeoPixel(board.D18, 55, brightness=1)
def cleanup_led():
    pixels.fill((0, 0, 0))

atexit.register(cleanup_led)

class Camera:
    def __init__(self):
        self.mainui = loadUi('baybayin-kraken-app.ui')
        self.mainui.show()
        pixels.fill((255, 255, 255))
        
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        self.picam2.start()
        self.mainui.startStopCameraButton.clicked.connect(self.closeEvent)# exit button
        self.mainui.captureImageButton.clicked.connect(self.capture_image) #capture frame button
        self.mainui.transliterateImageButton.clicked.connect(self.transliterate_image) # transliterate button
        self.mainui.binarizeImageButton.clicked.connect(self.binarize_image) #binarize button

        self.mainui.resetCameraButton.clicked.connect(self.reset_camera) # reset button
        self.mainui.showOutputTextButton.clicked.connect(self.show_output_text) # show output button


        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                                "haarcascade_frontalface_default.xml")

        self.video_label = self.mainui.cameraLabel
        self.video_label.setScaledContents(True)
        self.video_label.setFixedSize(429, 329)
        self.mainui.cameraLayout.addWidget(self.video_label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_frame = None
        self.frozen = False   # ✅ Track freeze state


    def update_frame(self):
        frame = self.picam2.capture_array()
        self.current_frame = frame.copy()

        # Convert to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Display in QLabel
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))


    def capture_image(self):
        if self.current_frame is not None:
            filename = "test_image.png"
            cv2.imwrite(filename, self.current_frame)
            print(f"Image saved as {filename}")
            self.mainui.ocrOutputBox.setPlainText("Image captured!")

            # ✅ Freeze preview
            self.timer.stop()
            self.frozen = True
        else:
            print("No frame available to capture.")


    def binarize_image(self):
        image = Image.open("test_image.png")
        binarized_image = binarization.nlbin(image, low=5, high=25)
        binarized_image.save("test_binarized.png")
        self.mainui.ocrOutputBox.setPlainText("Image binarized!")

    def transliterate_image(self):
        import os
        os.environ["PYTHONUTF8"] = "1"  # force utf-8 mode for subprocess

        command = [
            "kraken", "-i", "test_binarized.png", "test.txt", "segment", "ocr",
            "-m", "baybayin_custom_dataset_edited.mlmodel_best.mlmodel"
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

    def reset_camera(self):
        """Resume live camera feed after capture."""
        if self.frozen:
            self.timer.start(30)
            self.frozen = False
            self.mainui.ocrOutputBox.setPlainText("Camera reset. Live preview resumed.")
    
    def show_output_text(self):
        """Load OCR result from test.txt into outputText box."""
        try:
            with open("test.txt", "r", encoding="utf-8") as f:
                text = f.read()
                self.mainui.outputText.setPlainText(text)
        except FileNotFoundError:
            self.mainui.outputText.setPlainText("Error: test.txt not found.")
        except Exception as e:
            self.mainui.outputText.setPlainText(f"Error reading test.txt:\n{str(e)}")

    def closeEvent(self):
        try:
            self.picam2.stop()
        except:
            pass
        
        try:
            pixels.fill((0, 0, 0))
        except:
            pass
        
        subprocess.run(["sudo", "shutdown", "-h", "now"])
        if event:
            event.accept()

if __name__ == '__main__':
    app = QApplication([])
    main = Camera()
    app.exec()

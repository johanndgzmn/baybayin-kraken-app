from PyQt6.uic import loadUi
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

import cv2
import time
import subprocess
from PIL import Image
from kraken import binarization

class Camera:
    def __init__(self):
        self.mainui = loadUi('baybayin-kraken-app.ui')
        self.mainui.show()
        
        self.mainui.startStopCameraButton.clicked.connect(self.closeEvent)
        self.mainui.captureImageButton.clicked.connect(self.capture_image)
        self.mainui.transliterateImageButton.clicked.connect(self.transliterate_image)  # Connect transliterate button
        self.mainui.binarizeImageButton.clicked.connect(self.binarize_image)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            print("Error: Cannot access the webcam.")
            return
        
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

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                       minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
            self.video_label.repaint()

    def capture_image(self):
        if self.current_frame is not None:
            filename = "test_image.png"  # Fixed name for OCR input
            cv2.imwrite(filename, self.current_frame)
            print(f"Image saved as {filename}")
        else:
            print("No frame available to capture.")
        self.mainui.ocrOutputBox.setPlainText("Image captured!")

    def binarize_image(self):
        image = Image.open("test_image.png")
        binarized_image = binarization.nlbin(image, low=5, high=20)
        binarized_image.save("test_binarized.png")
        self.mainui.ocrOutputBox.setPlainText("Image binarized!")

    def transliterate_image(self):
        command = [
            "kraken", "-i", "test_binarized.png", "test.txt", "ocr",
            "-m", "baybayin_model_latin_text_v1.mlmodel_99.mlmodel", "--no-segmentation"
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output_text = result.stdout + "\n" + result.stderr
        except subprocess.CalledProcessError as e:
            output_text = "Error during OCR:\n" + e.stdout + "\n" + e.stderr

        # Display output in the text box
        print(output_text)

    def closeEvent(self):
        self.cap.release()
        self.mainui.close()

if __name__ == '__main__':
    app = QApplication([])
    main = Camera()
    app.exec()

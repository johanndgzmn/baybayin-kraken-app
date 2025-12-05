# Installation:
1. Clone repository
2. Download python 3.11 https://www.python.org/downloads/release/python-3110/
3. Create new virtual environment
```
py -3.11 -m venv kraken-env
kraken-env\Scripts\activate
```
4. Install dependencies
```
pip install kraken==5.3.0
pip install opencv-python
# pip install numpy==1.26.4 scipy==1.11.4
pip install pyqt6
# pip uninstall scikit-learn -y
# pip install scikit-learn==1.1.2
# pip uninstall torch -y
# pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install transformers sentencepiece accelerate
```
5. Run ```python cvvideoimport.py``` in the terminal

# Versions
1. cvvideoimport_rpi.py - Only uses Kraken. No post-processing. Built for RPI.
2. cvvideoimport_t5_rpi_ver.py - Uses Kraken with T5 post-processing. Built for RPI.
3. cvvideoimport_t5.py - Uses Kraken with T5 post-processing. Built for running in desktop.
4. cvvideoimport.py - Only uses Kraken, no post-processing. Built for desktop.
5. baybayin-kraken-app.ui - For Thesis 1
6. baybayin-kraken-app_post_process.ui - For Thesis 2
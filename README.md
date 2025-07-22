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
pip install kraken==5.2.1
pip install opencv-python
pip install numpy==1.26.4 scipy==1.11.4
pip install pyqt6
pip uninstall scikit-learn -y
pip install scikit-learn==1.1.2
pip uninstall torch -y
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```
5. Run ```python cvvideoimport.py``` in the terminal
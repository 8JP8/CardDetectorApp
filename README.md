# Playing Card Detector App

This application utilizes OpenCV and a YOLOv4 pre-trained model from Roboflow to detect playing cards in images or video streams. It requires Python 3.10 and several dependencies listed below.

## Requirements

- Python 3.10
- opencv-python
- pyqt6
- numpy
- inference_sdk
- configparser

## Installation

1. Install Python 3.10 from [Python's official website](https://www.python.org/downloads/).
2. Install the required packages using pip:

```bash
pip install opencv-python pyqt6 numpy inference_sdk configparser
```

## Usage
1. Clone or download the repository to your local machine.
2. Navigate to the directory containing the CardDetectorProgram.py file.
3. Run the Python script using the following command:
```bash
python CardDetectorProgram.py
```
4. The application will launch, allowing you to either upload an image/video or use your webcam for real-time detection.
5. Once the detection is complete, the detected playing cards will be highlighted or labeled on the screen.

> [!NOTE]
> 
 - Ensure that you have a stable internet connection during the initial setup as the YOLOv4 pre-trained model will be downloaded from Roboflow.
> - For optimal performance, use a machine with a GPU.
> - If you encounter any issues, refer to the documentation of the individual packages or contact the developers for support.

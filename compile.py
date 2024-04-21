import sys
sys.setrecursionlimit(1000000)
from cx_Freeze import setup, Executable

# Dependencies (add any additional dependencies your app may have)
build_exe_options = {
    "packages": ["PyQt6", "cv2", "numpy", "inference_sdk", "configparser", "win32gui", "win32con", "torch", "torchvision", "yolov5", "seaborn"],
    "excludes": ["tkinter"],
    "include_files": [
        "images/",   # Include the entire images directory
        "themes/",   # Include the entire themes directory
        "config.ini",  # Include specific files
        "main.ui",
        "model.pt",
    ],
}

# Executable
exe = Executable(
    script="CardDetectorProgram.py",  # Replace with the name of your script
    base="Win32GUI",
    icon="images/icon.ico",  # Provide the path to your icon file
)

setup(
    name="CardDetectorProgram",
    version="1.0",
    description="App to detect cards and count the points of them.",
    options={"build_exe": build_exe_options},
    executables=[exe],
)
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QTableWidgetItem
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap, QIcon, QColor, QFont
from PyQt6.uic import loadUi
from PyQt6 import QtCore
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import configparser
import win32gui, win32con
import torch
import configparser
from PIL import Image

# Import attempt_load from yolov5 module
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))  # Add yolov5 directory to system path
from yolov5.models.experimental import attempt_load

class DetectionThread(QThread):
    #SIGNALS
    started_capture_signal = pyqtSignal(str)
    frame_update_signal = pyqtSignal(np.ndarray)
    disable_label_signal = pyqtSignal(str,float)  # Signal to disable label
    
    global threshold
    threshold = 0.60
    

    def __init__(self):
        super().__init__()

    def run(self):
        global offlinedetection
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), 'config.ini'))
        offlinedetection = config.get('detection', 'offlinedetection')

                    
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=config.get('roboflow', "api_key"))

        model_id = "playing-cards-ow27d/4"
        if offlinedetection:
            model_path = os.path.join(os.getcwd(), config.get('detection','model_path'))


        cap = cv2.VideoCapture(0)
        
        self.started_capture_signal.emit("Capture Running") #emit capture started signal
        
        confidence = []
        while True:
            ret, frame = cap.read() 
            
            if not offlinedetection:
                try: result = CLIENT.infer(frame, model_id=model_id)
                except:  pass
            else:
                model = attempt_load(model_path)
                # Load the pre-trained model
                #model = torch.load(model_path, map_location=torch.device('cpu'))  # Assuming "best.pt" is a PyTorch model file
                 # Convert the frame to PIL Image
                result = model(frame)
                print(result)
            predictions = result['predictions']

            for prediction in predictions:
                x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
                confidence = prediction['confidence']
                class_name = prediction['class']
                
                # Emit signal to disable label
                self.disable_label_signal.emit(class_name, confidence)

                cv2.rectangle(frame, (x-int(width/2), y-int(height/2)), (x + int(width/2), y + int(height/2)), (0, 255, 0), 2)
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y+height-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #self.frame_update_signal.emit(frame)
            # Display the frame
            cv2.imshow("Playing Cards Inference", frame)
            hwnd = win32gui.FindWindow(None, "Playing Cards Inference")
            icon_path = os.path.join(os.getcwd(), "images", "images", "icon_black.ico") #You need an external .ico file, in this case, next to .py file
            win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, win32gui.LoadImage(None, icon_path, win32con.IMAGE_ICON, 0, 0, win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.started_capture_signal.emit("Idle") #emit capture started signal
                break

        cap.release()
        cv2.destroyAllWindows()

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('main.ui', self)
        self.setWindowIcon(QIcon(os.path.join(os.getcwd(), 'images', 'icon.ico')))
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)   
        self.setWindowTitle("Automatic Card Detector")
        self.video_label = self.findChild(QLabel, 'video_label')
        self.thread = DetectionThread()
        #self.thread.frame_update_signal.connect(self.update_frame)
        self.thread.disable_label_signal.connect(self.BlackOutCard)
        self.thread.started_capture_signal.connect(self.UpdateStatus)
        self.thread.start()
        self.UpdateStatus("Starting Camera Capture Stream")
        self.dragging = False
        self.offset = None
        
        #EVENTS
        self.closeAppBtn.clicked.connect(self.close_app)
        self.maximizeRestoreAppBtn.clicked.connect(self.maximize_restore_app)
        self.minimizeAppBtn.clicked.connect(self.minimize_app)
        self.resetTopBtn.clicked.connect(self.ResetCards)
        self.settingsTopBtn.clicked.connect(self.OpenSettings)

        # Find labels with LB tag and connect the mousePressEvent
        for widget in self.findChildren(QLabel):
            if widget.objectName().startswith("LB_"):
                widget.mousePressEvent = lambda event, label=widget: self.toggle_label(label)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if (self.titleLabel.frameGeometry().contains(event.pos()) or self.iconLabel.frameGeometry().contains(event.pos()) or self.artLabel.frameGeometry().contains(event.pos())):
                self.dragging = True
                self.offset = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(event.globalPosition().toPoint() - self.offset)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.offset = None

    def toggle_label(self, label):
        # Disable the clicked label
        if label.isEnabled():
            label.setEnabled(False)
        else:
            label.setEnabled(True)

        # Assuming self.tableWidget is an instance of QTableWidget
        for row in range(self.tableWidget.rowCount()):
            item = self.tableWidget.verticalHeaderItem(row)
            if item and item.text() == label.objectName()[3:]:  # Check if the item exists and its text matches the object name
                # Update the value in the third column of the current row
                state_item = QTableWidgetItem("Not Found" if label.isEnabled() else "Found")
                confd_item = QTableWidgetItem("" if label.isEnabled() else "Manually Selected")
                state_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                confd_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tableWidget.setItem(row, 1, state_item)
                self.tableWidget.setItem(row, 2, confd_item)
                break
        
    def close_app(self):
        app.closeAllWindows()
        
    def minimize_app(self):
        self.showMinimized()

    def maximize_restore_app(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        
    def BlackOutCard(self, class_name, confidencevalue):
        # Iterate through all child widgets of the main windo
        detectionpositive = False
        if confidencevalue > threshold:
            detectionpositive = True
        for widget in self.findChildren(QLabel):
            # Check if the widget name matches the pattern "LB_" + class_name
            if widget.objectName() == "LB_" + class_name:
                # Disable the widget
                if detectionpositive:
                    widget.setEnabled(False)
                break  # Exit loop once the widget is found    
            
        for row in range(self.tableWidget.rowCount()):
            item = self.tableWidget.verticalHeaderItem(row)
            if item.text() == class_name:  # Check if the item exists and its text matches the name
                print (item.text())
                # Update the value in the third column of the current row
                if detectionpositive:
                    state = QTableWidgetItem("Found")
                    confd = QTableWidgetItem(str(confidencevalue))
                    state.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    confd.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.tableWidget.setItem(row, 1, state)
                    self.tableWidget.setItem(row, 2, confd)
                break  # Exit loop once the row is found
        self.UpdateConfidenceAverage()
    
    def UpdateConfidenceAverage(self):
        total_confidence = 0
        num_items = 0

        # Iterate over all rows in the second column of the tableWidget, starting from the second row
        for row in range(1, self.tableWidget.rowCount()):
            item = self.tableWidget.item(row, 2)  # Get item in the second column (index 2)
            if item is not None:
                try:
                    total_confidence += float(item.text())  # Assuming the item contains a numeric value
                    num_items += 1
                except:
                    pass


        # Calculate average confidence
        average_confidence = total_confidence / num_items if num_items > 0 else 0

        # Update value displayed in averageconfidenceLcd
        self.averageconfidence_Lcd.display(average_confidence)

                        
            
    def ResetCards(self):
        if not self.thread.isRunning():
            self.thread.start()
        # Iterate through all child widgets of the main window
        for widget in self.findChildren(QLabel):
            if widget.objectName().find("LB_") != -1:
                # Enable the widget if it's a QLabel
                widget.setEnabled(True)
            
    def UpdateStatus(self, message):
        self.statuslb.setText("Status: " + message)
            
    def OpenSettings(self):
        if self.stackedWidget.currentIndex() == 0:
            self.stackedWidget.setCurrentIndex(1)
        else:
            self.stackedWidget.setCurrentIndex(0)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    my_app = MyApp()
    my_app.show()
    sys.exit(app.exec())

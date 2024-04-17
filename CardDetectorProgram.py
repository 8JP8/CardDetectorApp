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
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), 'config.ini'))
                    
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=config.get('roboflow', "api_key"))

        model_id = "playing-cards-ow27d/4"

        cap = cv2.VideoCapture(0)
        
        self.started_capture_signal.emit("Capture Running") #emit capture started signal
        
        confidence = -1
        while True:
            ret, frame = cap.read() 
            
            result = CLIENT.infer(frame, model_id=model_id)
            predictions = result['predictions']

            for prediction in predictions:
                x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
                if (confidence != -1) : confidence = (confidence + prediction['confidence']) / 2 
                else: confidence = prediction['confidence']
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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.started_capture_signal.emit("Idle") #emit capture started signal
                break

        cap.release()
        cv2.destroyAllWindows()

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('main.ui', self)
        self.setWindowIcon(QIcon(os.path.join(os.getcwd(), 'images', 'cards', 'aces.png')))
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
        self.titleLabel.linkActivated.connect(self.prints)
        self.settingsTopBtn.clicked.connect(self.OpenSettings)
        
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

    def prints(self):
        print("heasddasd")
        
    def close_app(self):
        app.closeAllWindows()
        
    def minimize_app(self):
        self.showMinimized()

    def maximize_restore_app(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        
    def BlackOutCard(self, class_name, confidence):
        # Iterate through all child widgets of the main windo
        detectionpositive = False
        if confidence > threshold:
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
                    confd = QTableWidgetItem(str(confidence))
                    state.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    confd.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.tableWidget.setItem(row, 1, state)
                    self.tableWidget.setItem(row, 2, confd)
                break  # Exit loop once the row is found
                
            
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

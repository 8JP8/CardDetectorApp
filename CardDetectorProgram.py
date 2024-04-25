import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QTableWidgetItem, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QIcon, QColor, QPainter, QPixmap
from PyQt6.uic import loadUi
from PyQt6 import QtCore
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import configparser
import win32gui, win32con
import torch
import configparser
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

class DetectionThread(QThread):
    #SIGNALS
    toggle_label_signal = pyqtSignal(QLabel)
    started_capture_signal = pyqtSignal(str)
    frame_update_signal = pyqtSignal(np.ndarray)
    warning_message_signal = pyqtSignal(str, str, str)
    disable_label_signal = pyqtSignal(str,float)  # Signal to disable label
    
    global card_names, lable_enabled_list
    card_names =  ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD', 'QH', 'QS']
    lable_enabled_list = []
    
    def scale_coords(self, img_shape, coords, actual_shape):
        gain = min(img_shape[0] / actual_shape[0], img_shape[1] / actual_shape[1])
        pad_x = (img_shape[1] - actual_shape[1] * gain) / 2
        pad_y = (img_shape[0] - actual_shape[0] * gain) / 2
        pad = (pad_x, pad_y)  # Convert tensor elements to float
        coords[:, :4] -= torch.tensor([pad[0], pad[1], pad[0], pad[1]], device=coords.device)  # Subtract padding from coordinates
        coords[:, :4] /= gain  # Scale coordinates
        coords[:, :4] = torch.clip(coords[:, :4], 0, img_shape[1])  # Clip coordinates to image boundaries
        return coords

    def __init__(self):
        super().__init__()

    def run(self):
        global offlinedetection
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), 'config.ini'))
        global threshold
        threshold = config.getfloat('detection', 'threshold')
        offlinedetection = config.get('detection', 'offlinedetection')

        if not offlinedetection:
            try:
                CLIENT = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key=config.get('roboflow', "api_key"))

                model_id = "playing-cards-ow27d/4"
            except: offlinedetection=True
        if offlinedetection:
            model_path = os.path.join(os.getcwd(), config.get('detection','model_path'))
            # Load the YOLOv5 model'
            use_cuda = config.getboolean('detection', 'model_use_cuda')
            if use_cuda:
                if torch.cuda.is_available():
                    if os.path.exists(os.path.join(os.getcwd(),config.get('detection','cuda_model_path'))):
                        model_path = os.path.join(os.getcwd(), config.get('detection','cuda_model_path'))
                    else: use_cuda = False
                else:
                    use_cuda = False
                    self.warning_message_signal.emit("<a href='https://pytorch.org/get-started/locally/'>Torch CUDA</a> not available (Falling back to CPU model)", "Warning", "Load Model - Warning")
            
            device = torch.device('cuda' if use_cuda else 'cpu')

            model = attempt_load(model_path)  # Ensure model is loaded to appropriate device
            model.eval()


        cap = cv2.VideoCapture(config.getint('detection','camera_index')) # Open a connection to the camera
        
        self.started_capture_signal.emit("Capture Running") #emit capture started signal
        confidence = []
        while True:
            ret, frame = cap.read() 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not offlinedetection:
                try: result = CLIENT.infer(frame, model_id=model_id)
                except:  pass
                predictions = result['predictions']
                for prediction in predictions:
                    if len(prediction) > 0:
                        x, y, width, height = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
                        confidence = prediction['confidence']
                        class_name = prediction['class']
                        
                        # Display the Rectangles
                        cv2.rectangle(frame, (x-int(width/2), y-int(height/2)), (x + int(width/2), y + int(height/2)), (0, 255, 0), 2)
                        text = f"{class_name} - {confidence:.2f}"
                        cv2.putText(frame, text, (x, y+height-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print(f'Card: {class_name} - Confidence: {confidence} - Bounding Box: {(x-width/2, y-height/2, x+width/2, y+height/2)}')

                        # Emit signal to disable label
                        self.disable_label_signal.emit(class_name, confidence)
                        
            else:
                 # Run inference
                with torch.no_grad():
                    img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    result = model(img)[0]
                    result = non_max_suppression(result, conf_thres=0.5, iou_thres=0.45)[0]
                    # Process output
                    if result is not None:
                        result[:, :4] = self.scale_coords(img.shape[2:], result[:, :4], frame.shape).round()
                    for x1, y1, x2, y2, confidence, cls_pred in result:
                        class_name = card_names[int(cls_pred)]
                        print(f'Card: {class_name} [{int(cls_pred)}] - Confidence: {confidence} - Bounding Box: {(int(x1), int(y1), int(x2), int(y2))}')

                        # Draw bounding box on the frame
                        text = f"{class_name} - {confidence:.2f}"
                        cv2.putText(frame, text, (int(x1), int(y1)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Emit signal to disable label
                        self.disable_label_signal.emit(class_name, confidence)

            # Display the frame
            viewframe = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Playing Cards Inference", viewframe)
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
        self.thread.toggle_label_signal.connect(self.toggle_label)
        self.thread.warning_message_signal.connect(self.message_box_warning)
        self.thread.start()
        self.dragging = False
        self.offset = None
        self.pixmaplist = []
        self.points = 0
        self.points_sum = 0
        self.spades_sum = 0
        self.hearts_sum = 0
        self.clubs_sum = 0
        self.diamonds_sum = 0
        self.UpdateStatus("Starting Camera Capture Stream")
        
        #EVENTS
        self.closeAppBtn.clicked.connect(self.close_app)
        self.maximizeRestoreAppBtn.clicked.connect(self.maximize_restore_app)
        self.minimizeAppBtn.clicked.connect(self.minimize_app)
        self.resetTopBtn.clicked.connect(self.ResetCards)
        self.settingsTopBtn.clicked.connect(self.OpenSettings)


        # Find labels with LB tag and connect the mousePressEvent
        for widget in self.findChildren(QLabel):
            if widget.objectName().startswith("LB_"):
                self.pixmaplist.append((widget.objectName(), widget.pixmap()))
                lable_enabled_list.append((widget.objectName(), True))
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
        # Find the label in the list
        for index, (lb, enabled) in enumerate(lable_enabled_list):
            if label.objectName() == lb:
                # Toggle the enabled state
                if not enabled: 
                    self.apply_brightness_filter(label, False)
                    lable_enabled_list[index] = (lable_enabled_list[index][0], True)
                    self.CountPoints(lb[3:], False)
                else:
                    self.apply_brightness_filter(label, True)
                    lable_enabled_list[index] = (lable_enabled_list[index][0], False)
                    self.CountPoints(lb[3:])
                    
                self.CountPoints(lb)
                    
                # Assuming self.tableWidget is an instance of QTableWidget
                for row in range(self.tableWidget.rowCount()):
                    item = self.tableWidget.verticalHeaderItem(row)
                    if item and item.text() == label.objectName()[3:]:  # Check if the item exists and its text matches the object name
                        # Update the value in the third column of the current row
                        state_item = QTableWidgetItem("Not Found" if not enabled else "Found")
                        confd_item = QTableWidgetItem("" if not enabled else "Manually Selected")
                        state_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        confd_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.tableWidget.setItem(row, 1, state_item)
                        self.tableWidget.setItem(row, 2, confd_item)
                        break
                    
                    
    def CountPoints(self, card, sum=True):
        for row in range(self.tableWidget.rowCount()):
            if self.tableWidget.verticalHeaderItem(row).text() == card:
                if card.startswith("A"): self.points = 11
                elif card.startswith("7"): self.points = 10
                elif card.startswith("K"): self.points = 4
                elif card.startswith("J"): self.points = 3
                elif card.startswith("Q"): self.points = 2
                else: self.points = 0

                if sum:
                    if card.endswith("S"): self.spades_sum += self.points
                    if card.endswith("H"): self.hearts_sum += self.points
                    if card.endswith("C"): self.clubs_sum += self.points
                    if card.endswith("D"): self.diamonds_sum += self.points
                    self.points_sum += self.points
                else:
                    if card.endswith("S"): self.spades_sum -= self.points
                    if card.endswith("H"): self.hearts_sum -= self.points
                    if card.endswith("C"): self.clubs_sum -= self.points
                    if card.endswith("D"): self.diamonds_sum -= self.points
                    self.points_sum -= self.points
            
        #UI Updating
        self.total_sum_Lcd.display(self.points_sum)
        self.spades_sum_Lcd.display(self.spades_sum)
        self.clubs_sum_Lcd.display(self.clubs_sum)
        self.diamonds_sum_Lcd.display(self.diamonds_sum)
        if "Idle" in self.statuslb.text(): self.UpdateStatus("Idle")
        elif "Starting Camera Capture Stream" in self.statuslb.text(): self.UpdateStatus("Starting Camera Capture Stream")
        elif "Capture Started" in self.statuslb.text(): self.UpdateStatus("Capture Started")
    
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
                    #widget.setEnabled(False)
                    for index, (lb, enabled) in enumerate(lable_enabled_list):
                        if lb == widget.objectName() and enabled:
                            lable_enabled_list[index] = (lable_enabled_list[index][0], False)
                            self.apply_brightness_filter(widget, True)
                            self.CountPoints(class_name)
                            break
                break  # Exit loop once the widget is found    
            
        for row in range(self.tableWidget.rowCount()):
            item = self.tableWidget.verticalHeaderItem(row)
            if item.text() == class_name:  # Check if the item exists and its text matches the name
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
        self.UpdateStatus("Capture Running")
    
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
                #widget.setEnabled(True)
                self.apply_brightness_filter(widget, False)
        for index, (lb, enabled) in enumerate(lable_enabled_list):
            lable_enabled_list[index] = (lable_enabled_list[index][0], True)
        self.points_sum = 0
        self.spades_sum = 0
        self.clubs_sum = 0
        self.diamonds_sum = 0
        self.total_sum_Lcd.display(self.points_sum)
        self.spades_sum_Lcd.display(self.spades_sum)
        self.clubs_sum_Lcd.display(self.clubs_sum)
        self.diamonds_sum_Lcd.display(self.diamonds_sum)
        self.averageconfidence_Lcd.display(0)
        self.UpdateStatus("Starting Camera Capture Stream")
        for row in range(1, self.tableWidget.rowCount()):
            # Replace values in the second column with "Not Found"
            item = QTableWidgetItem("Not Found")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tableWidget.setItem(row, 1, item)
            # Clear values in the third column
            self.tableWidget.setItem(row, 2, None)

            
    def UpdateStatus(self, message):
        if self.points_sum != 0:
            message += f" - {self.points_sum}"
        self.statuslb.setText("Status: " + message)
            
    def OpenSettings(self):
        if self.main_stackedwidget.currentIndex() == 0:
            self.main_stackedwidget.setCurrentIndex(1)
        else:
            self.main_stackedwidget.setCurrentIndex(0)
            
    def apply_brightness_filter(self, label, apply):
        if apply:
            pixmap = label.pixmap()
            if pixmap.isNull():
                return
            
            width = pixmap.width()
            height = pixmap.height()

            # Create a new pixmap with the same size as the original pixmap
            modified_pixmap = QPixmap(width, height)
            modified_pixmap.fill(QColor("transparent"))

            # Apply brightness adjustment
            painter = QPainter(modified_pixmap)
            painter.setOpacity(0.5)
            painter.drawPixmap(0, 0, pixmap)
            painter.end()

            # Update the label with the modified pixmap
            label.setPixmap(modified_pixmap)
        else:
            for name, o_pixmap in self.pixmaplist:
                if name == label.objectName():
                    label.setPixmap(o_pixmap)

    def message_box_warning(self, message, type, title):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setWindowIcon(QIcon(os.path.join(os.getcwd(), "images", "images", "icon_black.ico")))
        msg_box.setTextFormat(Qt.TextFormat.RichText)  # Set text format to RichText
        if type == "Error":
            msg_box.setIcon(QMessageBox.Icon.Critical)
        elif type == "Warning":
            msg_box.setIcon(QMessageBox.Icon.Warning)
        elif type == "Info":
            msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setText(message)
        msg_box.exec()
            
    
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    my_app = MyApp()
    my_app.show()
    sys.exit(app.exec())

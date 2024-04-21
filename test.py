import torch
import cv2
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def scale_coords(img_shape, coords, actual_shape):
    gain = min(img_shape[0] / actual_shape[0], img_shape[1] / actual_shape[1])
    pad_x = (img_shape[1] - actual_shape[1] * gain) / 2
    pad_y = (img_shape[0] - actual_shape[0] * gain) / 2
    pad = (pad_x, pad_y)  # Convert tensor elements to float
    coords[:, :4] -= torch.tensor([pad[0], pad[1], pad[0], pad[1]], device=coords.device)  # Subtract padding from coordinates
    coords[:, :4] /= gain  # Scale coordinates
    coords[:, :4] = torch.clip(coords[:, :4], 0, img_shape[1])  # Clip coordinates to image boundaries
    return coords


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# Load the YOLOv5 model
weights_path = 'model.pt'
device = torch.device("""'cuda' if torch.cuda.is_available() else""" 'cpu')
model = attempt_load(weights_path)  # Ensure model is loaded to appropriate device
model.eval()

# Open a connection to the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the input frame
    img_size = 640  # You may need to adjust this based on your training configuration
    img, _, _ = letterbox(frame, new_shape=(img_size, img_size))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Run inference
    with torch.no_grad():
        prediction = model(img)[0]
        prediction = non_max_suppression(prediction, conf_thres=0.5, iou_thres=0.45)[0]

    # Process output
    if prediction is not None:
        prediction[:, :4] = scale_coords(img.shape[2:], prediction[:, :4], frame.shape).round()
        print(prediction)
        for x1, y1, x2, y2, conf, cls_pred in prediction:
            print(f'Object: {int(cls_pred)} - Confidence: {conf.item()} - Bounding Box: {(x1, y1, x2, y2)}')

            # Draw bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import pyautogui
import cv2
from PIL import ImageGrab
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time
import torch
import pydirectinput
torch.cuda.set_device(0)
def get_color_for_class(class_index):
    class_colors = [
        (255, 0, 0),   # Red for class 0
        (0, 255, 0),   # Green for class 1
        (0, 0, 255)    # Blue for class 2
        # Add more class-color mappings as needed
    ]
    return class_colors[class_index] if 0 <= class_index < len(class_colors) else (128, 128, 128)  # Default to grey if class index is out of range


yolo_model = YOLO('sugibronze.pt')
try:
    while True:
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        # Use YOLOv8 to perform object detection
        detections = yolo_model.predict(frame, stream=True)

        for r in detections:

            annotator = Annotator(frame)
            boxes = r.boxes

            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                cc = box.conf.item()

                if cc >= 0.7:
                    x_center = int((b[0] + b[2]) / 2)
                    y_center = int((b[1] + b[3]) / 2)
                    local = (x_center, y_center)

                    label = f"{yolo_model.names[int(c)]} - Confidence: {cc:.2f} - Local: {local}"
                    color = get_color_for_class(int(c))
                elif c == 2:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'I SHOULD BE PRESSING 1 AND +', (100, 100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                    pydirectinput.press('+')
                    pydirectinput.press('1')
            annotator.box_label(b, label, color=color)
            img = annotator.result()
            imS = cv2.resize(img, (960, 540))
        cv2.imshow('YOLO V8 Detection', imS)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
            cv2.destroyAllWindows()

except KeyboardInterrupt:
    pass

print('it broke')


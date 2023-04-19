from ultralytics import YOLO
import cv2
import numpy as np
from datetime import timedelta
import os
import base64
import torch
from supervision.tools.detections import Detections, BoxAnnotator

box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)

def detect_track(frame):
    model = YOLO('best.pt')
    CLASS_NAMES_DICT = model.model.names
    results = model(frame)
    if results.boxes.id is not None:
          detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    # format custom labels
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    # annotate and display frame    
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    cv2.imwrite("out.jpg", )


video_path = os.path.join('mask_main.mp4')
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
detect_track(frame)
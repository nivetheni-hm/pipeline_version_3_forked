# Importing required libraries
# from IPython import display
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from datetime import timedelta
import os
# from IPython.display import HTML
import base64
# from IPython.display import Image
# import torch.multiprocessing as mp
import torch

# display.clear_output()

def get_video_info(video_path):
    
    # Extracting information about the video
    video_info = sv.VideoInfo.from_video_path(video_path)
    width, height, fps, total_frames = video_info.width, video_info.height, video_info.fps, video_info.total_frames
    
    # Calculating the length of the video by dividing the total number of frames by the frame rate and rounding to the nearest second
    video_length = timedelta(seconds = round(total_frames / fps))
    
    # Print out the video resolution, fps, and length u
    print(f"\033[1mVideo Resolution:\033[0m ({width}, {height})")
    print(f"\033[1mFPS:\033[0m {fps}")
    print(f"\033[1mLength:\033[0m {video_length}")

get_video_info('/home/nivetheni/TCI_express/sample_vid5.mp4')

def track_yolo(source_path, destination_path, line_start, line_end):
  cap = cv2.VideoCapture(source_path)
  ret, frame = cap.read()
  # frame = cv2.imread("/home/nivetheni/TCI_express/warehouse_img1.png")
  # Load the pre-trained YOLO model
  model = YOLO('best_act.pt')

  # Create two points from the line_start and line_end coordinates
  line_start = sv.Point(line_start[0], line_start[1])
  line_end = sv.Point(line_end[0], line_end[1])
  
  # Create a line zone object from the two points
  line_counter = sv.LineZone(line_start, line_end) 
  
  # Create a line zone annotator object with specific thickness and text scale
  line_annotator = sv.LineZoneAnnotator(thickness = 2,
                                        text_thickness = 1,
                                        text_scale = 0.5)
  
  # Create a box annotator object with specific thickness and text scale
  box_annotator = sv.BoxAnnotator(thickness = 1,
                                  text_thickness = 1,
                                  text_scale = 0.5)
  
  # Extract information about the video from the given source path   
  video_info = sv.VideoInfo.from_video_path(source_path)

  # Create a video out path by combining the destination path and the video name
  video_name = os.path.splitext(os.path.basename(source_path))[0] + ".mp4"
  video_out_path = os.path.join(destination_path, video_name)

  # Create a video writer object for the output video
  video_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), video_info.fps, (video_info.width, video_info.height))
  
  # Loop over each frame of the video and perform object detection and tracking
  #  for result in model.track(source = source_path, tracker = 'bytetrack.yaml', show=False, stream=False, agnostic_nms=True):
  result = model(frame)
  # Get the original frame from the detection result
  # print(result)
  # print(dir(result))
  # print(dir(result[0]))

  frame = result[0].orig_img

  # Convert the YOLO detection results to a Detections object
  detections = sv.Detections.from_yolov8(result[0])

  # If the detections have an associated ID, set the tracker ID in the Detections object
  if result[0].boxes.id is not None:
    detections.tracker_id = result[0].boxes.id.cpu().numpy().astype(int)

  # Filter the detections to only include classes 2 (cars) and 7 (trucks)
  #detections = detections[(detections.class_id == 0) | (detections.class_id == 1)]
# print(detections)
# # Generate labels for the detections, including the tracker ID, class name, and confidence
# labels = [f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
#           for _, confidence, class_id, tracker_id 
#           in detections]
  labels = []

  for i in range(len(detections)):
  # print(detections.class_id, detections.tracker_id)
    if detections.tracker_id is not None and detections.class_id is not None:
      labels.append(str(detections.tracker_id[i]) + " " +str(model.model.names[detections.class_id[i]]))

  line_counter.trigger(detections)

# Annotate the frame with the line zone and any intersecting detections
  line_annotator.annotate(frame, line_counter)

# Annotate the frame with bounding boxes and labels for all detections
# frame = box_annotator.annotate(scene = frame,
#                                detections = detections,
#                                labels = labels)
  frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)
  cv2.imwrite("out.jpg", frame)
      # Write the annotated frame to the output video
      #video_out.write(frame)

  # Release the video writer and clear the Jupyter Notebook output
  video_out.release()

num_process = 4
torch.set_num_threads(1)
processes = []
video_path = os.path.join('/home/nivetheni/TCI_express/sample_vid5.mp4')

# while True:
track_yolo(source_path = "/home/nivetheni/TCI_express/sample_vid5.mp4",
              destination_path = "./res",
              line_start = (0,0),
              line_end = (0,0))

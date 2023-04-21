from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from datetime import timedelta
from pathlib import Path
import os
import torch
from PIL import Image
from pathlib import Path

from anamoly import anamoly_score_calculator, frame_weighted_avg

activity_model = YOLO('best_act.pt')
object_model = YOLO('yolov8_three_class.pt')

frame_cnt = 0 
batch_data = []
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def save_one_box(xyxy, im, file=Path('im.jpg'), gain=1.02, pad=50, square=False, BGR=False, save=True):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix('.jpg'))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop

def track_yolo(im2):
        global frame_cnt
        activity_results = activity_model.track(source=im2,tracker = 'botsort.yaml',persist=True)
        frame_cnt = frame_cnt + 1
        # if frame_cnt % 10 == 0:
        #     cv2.imwrite("/home/nivetheni/TCI_express/out1/"+str(frame_cnt)+".jpg",activity_results[0].plot())
        clssdict = activity_results[0].names
        frame_data = []
    
        conf_list = [round(each,3) for each in activity_results[0].boxes.conf.tolist()]
        #creating required lists form detection results only if it has tracking id 
        if  activity_results[0].boxes.is_track and len(conf_list) > 0:
            
            id_list = activity_results[0].boxes.id.tolist()
            class_list = [clssdict[each] for each in activity_results[0].boxes.cls.tolist()]
            bbox_list = activity_results[0].boxes.xyxy.tolist()

            #create crops and create cid for crop img
            crops = []
            for box in bbox_list:
                crop = save_one_box(box, im2, save=False)
                crops.append([crop])
                 
            #filter the generated lists 
            for i in range(0,len(conf_list)):
                if conf_list[i] < 0.50:
                    conf_list.pop(i)
                    id_list.pop(i)
                    class_list.pop(i)
                    bbox_list.pop(i)
        
            #create list of detections list for each frame
            for i in range(0,len(id_list)):
                detect_dict = {id_list[i]:{'type': "Person", 'activity': class_list[i],"confidence":conf_list[i],"crops":crops[i]}}
                frame_data.append(detect_dict)

            frame_info_anamoly = anamoly_score_calculator(frame_data)
            print(frame_info_anamoly)
            frame_anamoly_wgt = frame_weighted_avg(frame_info_anamoly)
            print(frame_anamoly_wgt)

            final_frame = {"frame_id":frame_cnt,"frame_anamoly_wgt":frame_anamoly_wgt,"detection_info":frame_info_anamoly,"cid":activity_results[0].plot()}
            print(final_frame)
            # # batching the frames
            # if frame_info_anamoly != []:
            #     batch_data.append(final_frame)
            #     if len(batch_data)==30:
            #           ##call json structuring   


        # else:
        #     print(activity_results[0].boxes)
        

# dir_list = os.listdir("/home/nivetheni/TCI_express/test_data")
# input_array = [] 
# for each in dir_list:
#     arr  =  cv2.imread("/home/nivetheni/TCI_express/test_data/"+each)
#     track_yolo(arr)
#     # input_array.append(arr)

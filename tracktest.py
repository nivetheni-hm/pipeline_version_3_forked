from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from datetime import timedelta
import os
import base64
import torch
from PIL import Image
import time
import datetime
from pathlib import Path
model = YOLO('yolov8_three_class.pt')
activity_model = YOLO("best_act.pt")
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

# def get_ppl_dict():

# cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

def plot_bbox(bbox_list,conf_list,id_list,class_list,im2):
    idx = 0
    result = im2.copy()
    for box in bbox_list:
        text = str(id_list[idx])+" "+"person"+" "+str(class_list[idx])+" "+str(conf_list[idx])
        cv2.rectangle(result, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(result, text, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        idx=idx+1
    return result

def track_yolo_arrin(input_array):


    for i,im2 in enumerate(input_array):
        activity_results = activity_model.track(source=im2,tracker = 'bytetrack.yaml',persist=True)
        conf_list = [round(each,3) for each in activity_results[0].boxes.conf.tolist()]
        if  activity_results[0].boxes.is_track and len(conf_list) > 0:
            clssdict = activity_results[0].names
            
            id_list = activity_results[0].boxes.id.tolist()
            class_list = [clssdict[each] for each in activity_results[0].boxes.cls.tolist()]
            bbox_list = activity_results[0].boxes.xyxy.tolist()
            inferenced_im2 = plot_bbox(bbox_list,conf_list,id_list,class_list,im2)
            cv2.imwrite("plot.jpg",inferenced_im2)
            time.sleep(15)
        # print(clssdict)
        # frame_data = []
        # #creating required lists form detection results only if it has tracking id 
        # if  activity_results[0].boxes.is_track:
        #     conf_list = [round(each,3) for each in activity_results[0].boxes.conf.tolist()]
        #     id_list = activity_results[0].boxes.id.tolist()
        #     class_list = [clssdict[each] for each in activity_results[0].boxes.cls.tolist()]
        #     bbox_list = activity_results[0].boxes.xyxy.tolist()

        #     #create crops and create cid for crop img
        #     crops = []
        #     for box in bbox_list:
        #         crop = save_one_box(box, im2, save=False)
        #         crops.append([crop])
                 
        #     #filter the generated lists 
        #     for i in range(0,len(conf_list)):
        #         if conf_list[i] < 0.50:
        #             conf_list.pop(i)
        #             id_list.pop(i)
        #             class_list.pop(i)
        #             bbox_list.pop(i)
        
        #     #create list of detections list for each frame
        #     for i in range(0,len(id_list)):
        #         detect_dict = {id_list[i]:{'type': "Person", 'activity': class_list[i],"confidence":conf_list[i],"crops":crops[i]}}
        #         frame_data.append(detect_dict)
        # print(frame_data)
        


        # if activity_results[0].boxes.is_track:
        #     conf_lst = activity_results[0].boxes.conf.tolist()

    #     results = model.track(source=im2,tracker = 'bytetrack.yaml', persist=True, stream=True)
    #     clssdict = results[0].names
    #     if results[0].boxes.is_track:
    #         conf_lst = results[0].boxes.conf.tolist()
    #         index_list = []

    #         cls_lst = [clssdict[each] for each in results[0].boxes.cls.tolist()]
    #         for i in len(cls_lst):
    #             if cls_lst[i] == "Person":
    #                 index_list.append(i)

    #         for i in index_list:
    #             if conf_lst[i] < 50:
    #                 index_list.pop(i)

    #         id_lst = results[0].boxes.id.tolist()
    #         bbx_lst = results[0].boxes.xyxy.tolist()

    #         for idx in index_list:
    #             activity_results = model.track(source=save_one_box(bbx_lst[idx], im2, save=False), persist=True)
    #             act_clssdict = activity_results[0].names
    #             activity_class = [act_clssdict[each] for each in activity_results[0].boxes.cls.tolist()][0]
                

                # people = {id_lst[idx]:{'type': cls_lst[idx], 'activity': labells,"confidence":conf_lst[idx],"crops":cidd}}

                  
            


            # for i,obj_id in enumerate(id_lst):
            #     people = {cd: {'type': detect_obj, 'activity': labells,"confidence":0,"did":did,"track_type":track_type,"crops":cidd}}




        # print(dir(results[0]))

    #     confidence = results[0].boxes.conf.tolist()
    #     boxes = results[0].boxes.xyxy.tolist()
    #     print("confidence: ",confidence)
    #     print("boxes: ",boxes)
    #     for i in len(boxes):
    #     crop_coods = results[0].boxes.xyxy.tolist()[0]
    #     # crop_coods = [each+each/10 for each in results[0].boxes.xyxy.tolist()[0]]
    #     crop_img = save_one_box(crop_coods, im2, save=False)

    #     cv2.imwrite(str(i)+".jpg",results[0].plot())
    #     # break
    # # end = time.time()

    return True

def track_yolo(pathh):
    # im1 = Image.open(pathh).convert('RGB')
    # results = model.track(source = im1, persist=True)
    # res_plotted = results[0].plot()
    # im = Image.fromarray(res_plotted)
    # im.save("/home/nivetheni/TCI_express/out.png")
    # time.sleep(15)

    im2 = cv2.imread(pathh)
    # im2 = Image.open('03.png').convert('RGB')
    results = model.track(source=im2, persist=True)
    res_plotted = results[0].plot()
    cv2.imwrite("/home/nivetheni/TCI_express/out.png",res_plotted)


# dir_list = os.listdir("/home/nivetheni/TCI_express/test_data")
# for each in dir_list:
#     track_yolo("/home/nivetheni/TCI_express/test_data/"+each)

dir_list = os.listdir("/home/nivetheni/TCI_express/test_data")
input_array = [] 
for each in dir_list:
    arr  =  cv2.imread("/home/nivetheni/TCI_express/test_data/"+each)
    input_array.append(arr)
# print(input_array)
# input = input_array + input_array + input_array + input_array + input_array + input_array + input_array + input_array

track_yolo_arrin(input_array)

# print(len(input))

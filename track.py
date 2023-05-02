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
import glob
from nanoid import generate
import psutil
import asyncio
import subprocess as sp
import ast
import json

from anamoly import anamoly_score_calculator, frame_weighted_avg
from json_converter import output_func
from person_type import find_person_type

from concurrent.futures import ThreadPoolExecutor

# Nats
from nats.aio.client import Client as NATS
import nats

activity_model = YOLO('best27_4_23.pt')
object_model = YOLO('yolov8_three_class.pt')

frame_cnt = 0 
final_batch = []
batch_data = []
veh_pub = True
dictt = {}
nats_urls = os.getenv("nats")
nats_urls = ast.literal_eval(nats_urls)

# from main import get_info

# data_info = get_info()

async def json_publish_activity(primary):    
    nc = await nats.connect(servers=nats_urls , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    JSONEncoder = json.dumps(primary)
    json_encoded = JSONEncoder.encode()
    Subject = "service.activities"
    Stream_name = "services"
    ack = await js.publish(Subject, json_encoded)
    print(" ")
    print(f'Ack: stream={ack.stream}, sequence={ack.seq}')
    print("Activity is getting published")

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

def plot_bbox(bbox_list,conf_list,id_list,class_list,im2):
    idx = 0
    result = im2.copy()
    for box in bbox_list:
        text = str(id_list[idx])+" "+"person"+" "+str(class_list[idx])+" "+str(conf_list[idx])
        cv2.rectangle(result, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.putText(result, text, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        idx=idx+1
    return result

async def inference(inn):
    # print(len(inn))
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(activity_model,inn)
        print(results)
        for result in results:
            print(result[0].boxes.id)
        print("--------------------------------------")

def track_yolo(im2, device_data, datainfo):
    # print(device_data[0])
    # print(im2.shape)
    global veh_pub,dictt,frame_cnt
    # print(len(dictt))
    # print(dictt.keys())
    if device_data[0] not in dictt:
        print("1st")
        dictt[device_data[0]] = []
        dictt[device_data[0]].append(im2)
    else:
        dictt[device_data[0]].append(im2)

    # asyncio.run(inference(im2))
    

    if len(dictt) == 2 and 0 not in [len(dictt[each]) for each in dictt]:
        inn = []
        for each in dictt:
            inn.append(dictt[each].pop(0))
        asyncio.run(inference(inn)) 
        # inference(inn)




    # activity_results = activity_model.track(source=im2,tracker = 'bytetrack.yaml',persist=False)

    # frame_cnt = frame_cnt + 1

    # clssdict = activity_results[0].names
    # frame_data = []
    
    # conf_list = [round(each,3) for each in activity_results[0].boxes.conf.tolist()]

    # if  activity_results[0].boxes.is_track and len(conf_list) > 0:
        
    #     id_list = activity_results[0].boxes.id.tolist()
    #     class_list = [clssdict[each] for each in activity_results[0].boxes.cls.tolist()]
    #     bbox_list = activity_results[0].boxes.xyxy.tolist()

    #     #create crops 
    #     crops = []
    #     for box in bbox_list:
    #         crop = save_one_box(box, im2, save=False)
    #         crops.append([crop])
        
    #     # print(conf_list)
    #     conf_list1 = conf_list
    #     id_list1 = []
    #     class_list1 = []
    #     bbox_list1 = []
    #     conf_list = []
    #     crops1 = []
    #     #filter the generated lists 
    #     for i in range(0,len(conf_list1)):
    #         # print(conf_list1[i])
    #         if conf_list1[i] > 0.50:
    #             conf_list.append(conf_list1[i])
    #             id_list1.append(id_list[i])
    #             class_list1.append(class_list[i])
    #             bbox_list1.append(bbox_list[i])
    #             crops1.append(crops[i])

    #     id_list = id_list1
    #     class_list = class_list1
    #     bbox_list = bbox_list1
    #     crops = crops1
    #     #plots bbox for detections whose confidence is more than 0.50
    #     inferenced_im2 = plot_bbox(bbox_list,conf_list,id_list,class_list,im2)
    
    #     #create list of detections list for each frame
    #     for i in range(0,len(id_list)):
    #         # print("ENTERNING INTO CROPS")
    #         # cv2.imwrite("./crops/"+str(frame_cnt)+".jpg", (crops[i][0]))
    #         # did = find_person_type((crops[i][0]), datainfo)
    #         # print("DID: ", did)
    #         detect_dict = {id_list[i]:{'type': "Person", 'activity': class_list[i],"confidence":conf_list[i],"crops":crops[i]}}
    #         frame_data.append(detect_dict)

    #     frame_info_anamoly = anamoly_score_calculator(frame_data)
    #     # # print(frame_info_anamoly)
    #     frame_anamoly_wgt = frame_weighted_avg(frame_info_anamoly)
    #     # print(frame_anamoly_wgt)
    #     # cidd = None
    #     # if frame_info_anamoly != []

    #     final_frame = {"frame_id":frame_cnt,"frame_anamoly_wgt":frame_anamoly_wgt,"detection_info":frame_info_anamoly,"cid":[inferenced_im2]}
        
    #     # print("FINAL FRAME: ", final_frame)
        
    #     if final_frame is not None:
    #         if len(batch_data) != 30:
    #             batch_data.append(final_frame)
    #         else:
    #             batchId = generate(size=32)
    #             # final_batch = [batch_data]
    #             print("############################################################################################")
    #             print("FRAME COUNT:", frame_cnt)
    #             print("LENGTH: ", len(batch_data))
    #             # print("BATCH DATA:", final_batch)
    #             # output_func([batch_data])
    #             result = output_func([batch_data])
    #             result['deviceid'] = device_data[0]
    #             result['timestamp'] = device_data[2]
    #             result['geo'] = {"latitude":device_data[4], "longitude":device_data[5]}
    #             pid = psutil.Process()
    #             memory_bytes = pid.memory_info().rss
    #             memory_mb = memory_bytes / 1024 / 1024
    #             mbb = f"{memory_mb:.2f}"
    #             result['memory'] = str(float(mbb) / 1024) + " GB"
    #             result["version"] = "v0.0.2"   
    #             result["batchid"] = batchId

    #             if len(result['metaData']['anamolyIds'])>0:
    #                 result['type']='anamoly'
                
    #             if result['metaData']['count']['peopleCount'] == 0 or result['metaData']['count']['vehicleCount'] != 0:
    #                 veh_pub =True
                
    #             if result['metaData']['count']['peopleCount'] != 0 or result['metaData']['count']['vehicleCount'] != 0:
                    
    #                 print(veh_pub)
    #                 if veh_pub:
    #                     # print(outt_)
    #                     asyncio.run(json_publish_activity(primary=result))
    #                     print(result)
    #                     veh_pub = False
                        
    #             torch.cuda.empty_cache()
                
    #             print("RESULT: \n", result)
    #             print("############################################################################################")
    #             batch_data.clear()
    #             # final_batch.clear()

    #     if frame_cnt % 10 == 0:
    #         cv2.imwrite("./out1/"+str(frame_cnt)+".jpg",inferenced_im2)


# #unit testing track.py
# # Get the list of all files inside in folder
# for i in range(596,1897):
#     device_data = ['12b1d7c0-d066-11ed-83df-776209d52ccf', 'uuid:626d6410-d723-4dc8-o837-e943c0987dcb', '2023-04-22 16:43:32.540612', 'DETECTION', 12.52105, 75.975952]
#     print("./in1/image_"+str(i)+".jpg")
#     im2 = cv2.imread("./in1/image_"+str(i)+".jpg")
#     track_yolo(im2, device_data)

# im2 = cv2.imread("/home/nivetheni/TCI_express/in1/image_1261.jpg")
# track_yolo(im2)
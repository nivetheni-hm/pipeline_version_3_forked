# gstreamer python bindings
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib

# os
import sys
import os

# concurrency and multi-processing 
import asyncio
import multiprocessing
from multiprocessing import Process, Queue, Pool

# Nats
from nats.aio.client import Client as NATS
import nats
# json
import json
# datetime
from pytz import timezone
import time
from datetime import datetime 
import imageio
import subprocess as sp
import torch
import shutil
# cv
import numpy as np
import cv2
import io
import re

#.env vars loaded
from os.path import join, dirname
from dotenv import load_dotenv
import ast
import gc
import psutil
from nanoid import generate

# from pytorchvideo.models.hub import slowfast_r50_detection
# from yolo_slowfast.deep_sort.deep_sort import DeepSort
# from memory_profiler import profile

#to fetch data from postgres
from db_fetch import fetch_db
from db_fetch_members import fetch_db_mem
from db_push import push_db

from lmdb_list_gen import attendance_lmdb_known, attendance_lmdb_unknown
from facedatainsert_lmdb import add_member_to_lmdb
from track import track_yolo
# from anamoly_track import trackmain
# from project_1_update_ import output_func
# obj_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./three_class_05_dec.pt')

# obj_model = torch.hub.load('Detection', 'custom', path='./three_class_05_dec.pt', source='local',force_reload=True)
# deepsort_tracker = DeepSort("./yolo_slowfast/deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
# device = 'cuda'
# video_model = slowfast_r50_detection(True).eval().to(device)

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")
nats_urls = os.getenv("nats")
nats_urls = ast.literal_eval(nats_urls)

pg_url = os.getenv("pghost")
pgdb = os.getenv("pgdb")
pgport = os.getenv("pgport")
pguser = os.getenv("pguser")
pgpassword = os.getenv("pgpassword")

nc_client = NATS() # global Nats declaration
Gst.init(sys.argv) # Initializes Gstreamer, it's variables, paths

# Define some constants
WIDTH = 1920
HEIGHT = 1080
PIXEL_SIZE = 3
processes = []
queues = []

# creation of directories for file storage
hls_path = "./Hls_output"
gif_path = "./Gif_output"
    
if os.path.exists(hls_path) is False:
    os.mkdir(hls_path)
    
if os.path.exists(gif_path) is False:
    os.mkdir(gif_path)

# list variables
frames = []
numpy_frames = []
gif_frames = []
known_whitelist_faces = []
known_whitelist_id = []
known_blacklist_faces = []
known_blacklist_id = []
cid_unpin_cnt = 0
gif_cid_list = []
# flag variable
start_flag = False
image_count = 0
gif_batch = 0
batch_count = 0
frame_count = 0
track_type = []
veh_pub = True
only_vehicle_batch_cnt = 0
unique_device = []

def remove_cnts(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def load_lmdb_list():
    known_whitelist_faces1, known_whitelist_id1 = attendance_lmdb_known()
    known_blacklist_faces1, known_blacklist_id1 = attendance_lmdb_unknown()
    
    global known_whitelist_faces
    known_whitelist_faces = known_whitelist_faces1

    global known_whitelist_id
    known_whitelist_id = known_whitelist_id1
    
    global known_blacklist_faces
    known_blacklist_faces = known_blacklist_faces1

    global known_blacklist_id
    known_blacklist_id = known_blacklist_id1
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print(len(known_whitelist_faces), len(known_blacklist_faces))
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------")

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

# def activity_trackCall(source, device_data, datainfo):
#     global only_vehicle_batch_cnt,veh_pub
#     device_id = device_data[0]
#     device_urn = device_data[1]
#     timestampp = device_data[2]
#     lat = device_data[4]
#     long = device_data[5]
#     queue1 = Queue()
#     batchId = generate(size=32)

#     trackmain(
#         source, 
#         device_id, 
#         batchId,
#         queue1, 
#         datainfo,
#         obj_model,
#         deepsort_tracker,
#         video_model,
#         device
#         )

#     video_data = queue1.get()

#     # print(video_data)

#     outt_ = output_func(video_data)
#     outt_['deviceid'] = device_id
#     outt_['timestamp'] = timestampp
#     outt_['geo'] = {"latitude":lat, "longitude":long}
#     pid = psutil.Process()
#     memory_bytes = pid.memory_info().rss
#     memory_mb = memory_bytes / 1024 / 1024
#     mbb = f"{memory_mb:.2f}"
#     outt_['memory'] = str(float(mbb) / 1024) + " GB"
#     outt_["version"] = "v0.0.2"   
#     outt_["batchid"] = batchId

#     if outt_['metaData']['count']['peopleCount'] == 0 and outt_['metaData']['count']['vehicleCount'] > 0:
#         outt_['type']='tracking'
#     if len(outt_['metaData']['anamolyIds'])>0:
#         outt_['type']='anamoly'
#     print(" ")
#     print(" ")
#     print(" ")
#     print(" ")
#     # print(" ")
#     # print(outt_['metaData']['count']['peopleCount'],outt_['metaData']['count']['vehicleCount'])
#     # print(" ")
#     if outt_['metaData']['count']['peopleCount'] == 0 or outt_['metaData']['count']['vehicleCount'] != 0:
#         veh_pub =True


#     if outt_['metaData']['count']['peopleCount'] != 0 or outt_['metaData']['count']['vehicleCount'] != 0:

#         print(veh_pub)
#         if veh_pub:
#             # print(outt_)
#             asyncio.run(json_publish_activity(primary=outt_)) 
#             print(outt_)
#             veh_pub = False

#     torch.cuda.empty_cache()

#     pid = os.getpid()
#     print("killing ",str(pid))
#     sp.getoutput("kill -9 "+str(pid))

# def config_func(source, device_data,datainfo ):
    
#     # print("success")
#     device_id = device_data[0]
#     urn = device_data[1]
#     timestampp = device_data[2]
#     subsciptions = device_data[3]
#     lat = device_data[4]
#     long = device_data[5]
#     print(subsciptions)
    
#     activity_trackCall(source, device_data,datainfo)
#     # # activity_trackCall(source, device_data)
#     # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

def device_hls_push(device_id, device_info):
    hls_url = 'https://hls.ckdr.co.in/live/stream{device_id}/{device_id}.m3u8'.format(device_id = device_id)
    status = push_db(hls_url, device_id)
    return status

def numpy_creation(device_id, urn, img_arr, timestamp, device_data):
    print(device_id)
    # filename for mp4
    video_name_gif = gif_path + '/' + str(device_id)
    if not os.path.exists(video_name_gif):
        os.makedirs(video_name_gif, exist_ok=True)
        
    timestamp = re.sub(r'[^\w\s]','',timestamp)
    
    path = video_name_gif + '/' + str(timestamp).replace(' ','') + '.gif'
    
    global image_count, cid_unpin_cnt, gif_batch, gif_frames
    
    image_count += 1
    
    datainfo = [known_whitelist_faces, known_blacklist_faces, known_whitelist_id, known_blacklist_id]
    track_yolo(img_arr, device_data, datainfo)
    pid = os.getpid()
    # print(pid, device_id)
    # if (image_count < 31):
    #     numpy_frames.append(img_arr)
    #     gif_frames.append(img_arr)
    # elif (image_count >= 31):
    #     cv2.imwrite(device_id+".jpg",img_arr)
    #     # print(timestamp)

    #     datainfo = [known_whitelist_faces, known_blacklist_faces,known_whitelist_id,known_blacklist_id]
    #     # Process(target = config_func,args = (numpy_frames, device_data,datainfo,)).start()
    #     # config_func(numpy_frames, device_data,datainfo)
    #     gif_batch += 1 

        
    #     if gif_batch == 5:
    #         gif_frames = gif_frames[-100:]
    #         print(timestamp)
    #         print("Images added: ", len(gif_frames))
    #         print("Saving GIF file")
    #         with imageio.get_writer(path, mode="I") as writer:
    #             for idx, frame in enumerate(gif_frames):
    #                 print("Adding frame to GIF file: ", idx + 1)
    #                 writer.append_data(frame)
    #         queue.put(path)
    #         # print("PATH:", path)
    #         command = 'ipfs --api={ipfs_url} add {file_path} -Q'.format(ipfs_url=ipfs_url, file_path=path)
    #         gif_cid = sp.getoutput(command)
    #         # print(gif_cid)
            
    #         # os.remove(path)
    #         # print("The path has been removed")
    #         # await device_snap_pub(device_id = device_id, urn=urn, gif_cid = gif_cid, time_stamp = timestamp)
    #         # asyncio.run(device_snap_pub(device_id = device_id, urn=urn, gif_cid = gif_cid, time_stamp = timestamp))
            
    #         frames.clear()
    #         image_count = 0  
    #     image_count = 0
    #     numpy_frames.clear()


def gst_hls(device_id, device_info):
        
    location = device_info['rtsp'] # Fetching device info
    username = device_info['username']
    password = device_info['password']
    ddns_name = device_info['ddns']
    encode_type = device_info['videoEncodingInformation']
    
    print("Entering HLS Stream")
    
    # filename for hls
    video_name_hls = hls_path + '/' + str(device_id)
    if not os.path.exists(video_name_hls):
        os.makedirs(video_name_hls, exist_ok=True)
    print(video_name_hls)
    
    if(ddns_name == None):
        hostname = 'localhost'
    else:
        hostname = ddns_name
        
    try:
        if((encode_type.lower()) == "h264"):
            pipeline = Gst.parse_launch('rtspsrc name=h_rtspsrc_{device_id} location={location} latency=10 protocols="tcp" drop-on-latency=true user-id={username} user-pw={password} ! rtph264depay name=h_depay_{device_id} ! mpegtsmux name=h_mux_{device_id} ! hlssink name=h_sink_{device_id}'.format(location=location, device_id=device_id, username=username, password=password))
        elif((encode_type.lower()) == "h265"):
            pipeline = Gst.parse_launch('rtspsrc name=h_rtspsrc_{device_id} location={location} latency=10 protocols="tcp" drop-on-latency=true user-id={username} user-pw={password} ! rtph265depay name=h_depay_{device_id} ! mpegtsmux name=h_mux_{device_id} ! hlssink name=h_sink_{device_id}'.format(location=location, device_id=device_id, username=username, password=password))
        elif((encode_type.lower()) == "mp4"):
            pipeline = Gst.parse_launch('rtspsrc name=h_rtspsrc_{device_id} location={location} protocols="tcp" ! decodebin name=h_decode_{device_id} ! x264enc name=h_enc_{device_id} ! mpegtsmux name=h_mux_{device_id} ! hlssink name=h_sink_{device_id}'.format(device_id = device_id, location=location))
            
        # sink params
        sink = pipeline.get_by_name('h_sink_{device_id}'.format(device_id = device_id))

        # Location of the playlist to write
        sink.set_property('playlist-root', 'https://{hostname}/live/stream{device_id}'.format(device_id=device_id, hostname=hostname))
        # Location of the playlist to write
        sink.set_property('playlist-location', '{file_path}/{file_name}.m3u8'.format(file_path=video_name_hls, file_name=device_id))
        # Location of the file to write
        sink.set_property('location', '{file_path}/segment.%01d.ts'.format(file_path=video_name_hls))
        # The target duration in seconds of a segment/file. (0 - disabled, useful for management of segment duration by the streaming server)
        sink.set_property('target-duration', 10)
        # Length of HLS playlist. To allow players to conform to section 6.3.3 of the HLS specification, this should be at least 3. If set to 0, the playlist will be infinite.
        sink.set_property('playlist-length', 3)
        # Maximum number of files to keep on disk. Once the maximum is reached,old files start to be deleted to make room for new ones.
        sink.set_property('max-files', 6)
        
        if not sink or not pipeline:
            print("Not all elements could be created.")
        else:
            print("All elements are created and launched sucessfully!")

        # Start playing
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.SUCCESS:
            print("Successfully set the pipeline to the playing state.")
            
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set the pipeline to the playing state.")
            
    except TypeError as e:
        print(TypeError," gstreamer hls streaming error >> ", e)

def gst_mp4(dev):
    device_id, device_info = dev[0],dev[1]
    print(device_info)
    location = device_info['rtsp'] # Fetching device info
    username = device_info['username']
    password = device_info['password']
    subscriptions = device_info['subscriptions']
    encode_type = device_info['videoEncodingInformation']
    urn = device_info['urn']
    lat = device_info['lat']
    long = device_info['long']
    
    print("Entering Framewise Stream")
    

    def new_buffer(sink, device_id):
                     
        global image_arr
        device_data = []
        device_data.append(device_id)
        device_data.append(urn)
        device_data.append(datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f'))
        device_data.append(subscriptions)
        device_data.append(lat)
        device_data.append(long)
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            data = buffer.extract_dup(0, buffer.get_size())
            # Convert the bytes to a numpy array
            array = np.frombuffer(data, dtype=np.uint8)
            array = array.reshape((HEIGHT, WIDTH, PIXEL_SIZE))
            # filename = f'{device_id}/{Gst.util_get_timestamp()}.jpg'
            # os.makedirs(os.path.dirname(filename), exist_ok=True)
            # cv2.imwrite(filename, array)
            # queue.put(filename)
            datetime_ist = str(datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f'))
            numpy_creation(device_id=device_id, urn=urn, img_arr=array, timestamp=datetime_ist,device_data=device_data)    
        return Gst.FlowReturn.OK
    
    try:
        if((encode_type.lower()) == "h264"):
            pipeline = Gst.parse_launch('rtspsrc name=g_rtspsrc_{device_id} location={location} latency=200 protocols="tcp" user-id={username} user-pw={password} !  rtph264depay name=g_depay_{device_id} ! h264parse name=g_parse_{device_id} ! avdec_h264 name=h_decode_{device_id} ! videoconvert name=h_videoconvert_{device_id} ! videoscale name=h_videoscale_{device_id} ! video/x-raw,format=BGR,width=1920,height=1080,pixel-aspect-ratio=1/1,bpp=24 ! queue ! appsink name=g_sink_{device_id} sync=false max-buffers=1 drop=true'.format(location=location, device_id=device_id, username=username, password=password))
        elif((encode_type.lower()) == "h265"):
            pipeline = Gst.parse_launch('rtspsrc name=g_rtspsrc_{device_id} location={location} latency=200 protocols="tcp" user-id={username} user-pw={password} !  rtph265depay name=g_depay_{device_id} ! h265parse name=g_parse_{device_id} ! avdec_h265 name=h_decode_{device_id} ! videoconvert name=h_videoconvert_{device_id} ! videoscale name=h_videoscale_{device_id} ! video/x-raw,format=BGR,width=1920,height=1080,pixel-aspect-ratio=1/1,bpp=24 ! queue ! appsink name=g_sink_{device_id} sync=false max-buffers=1 drop=true'.format(location=location, device_id=device_id, username=username, password=password))
        elif((encode_type.lower()) == "mp4"):
            pipeline = Gst.parse_launch('rtspsrc name=g_rtspsrc_{device_id} location={location} protocols="tcp" ! decodebin name=g_decode_{device_id} ! videoconvert name=g_videoconvert_{device_id} ! videoscale name=g_videoscale_{device_id} ! video/x-raw,width=640, height=320 ! appsink name=g_sink_{device_id}'.format(location=location, device_id = device_id))
        if not pipeline:
            print("Not all elements could be created.")
        else:
            print("All elements are created and launched sucessfully!")
        
        # sink params
        sink = pipeline.get_by_name('g_sink_{device_id}'.format(device_id = device_id))
        
        sink.set_property("emit-signals", True)
        sink.connect("new-sample", new_buffer, device_id)
        
        # Start playing
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.SUCCESS:
            print("Able to set the pipeline to the playing state.")
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set the pipeline to the playing state.")
            
        GLib.MainLoop().run()
            
    except TypeError as e:
        print(TypeError," gstreamer Gif error >> ", e)  
             
#@profile
def call_gstreamer(device_data):
    print("Got device info from DB")
    devs = []
    for i,key in enumerate(device_data):
        # if key == "12b1d7c0-d066-11ed-83df-776209d52ccf":
        print(key)
        devs.append([key, device_data[key]])
    
    with Pool(len(devs)) as p:
        p.map(gst_mp4, devs)

async def device_info(msg):
    if msg.subject == "service.device_discovery":
        device_info = {}
        print("Received a Device data\n")  
        deviceInfo_raw = msg.data  # fetch data from msg
        print(deviceInfo_raw)
        deviceInfo_decode = deviceInfo_raw.decode("utf-8") # decode the data which is in bytes
        deviceInfo_json = json.loads(deviceInfo_decode) # load it as dict
        # print(deviceInfo_json)
        deviceInfo_username = deviceInfo_json['username'] # fetch all the individual fields from the dict
        deviceInfo_password = deviceInfo_json['password']
        deviceInfo_ip = deviceInfo_json['ddns']
        deviceInfo_port = deviceInfo_json['port']
        deviceInfo_rtsp = deviceInfo_json['rtsp']
        deviceInfo_encode = deviceInfo_json['videoEncodingInformation']
        deviceInfo_id = deviceInfo_json['deviceId']
        deviceInfo_urn = deviceInfo_json['urn']
        deviceInfo_sub = deviceInfo_json['subscriptions']
        lat = deviceInfo_json['lat']
        long = deviceInfo_json['long']
        device_info[deviceInfo_id] = {}
        device_info[deviceInfo_id]['urn'] = deviceInfo_urn
        device_info[deviceInfo_id]['videoEncodingInformation'] = deviceInfo_encode
        device_info[deviceInfo_id]['rtsp'] = deviceInfo_rtsp
        device_info[deviceInfo_id]['port'] = deviceInfo_port
        device_info[deviceInfo_id]['ddns'] = deviceInfo_ip
        device_info[deviceInfo_id]['password'] = deviceInfo_password
        device_info[deviceInfo_id]['username'] = deviceInfo_username
        device_info[deviceInfo_id]['subscriptions'] = deviceInfo_sub
        device_info[deviceInfo_id]['lat'] = lat
        device_info[deviceInfo_id]['long'] = long
        gst_mp4(deviceInfo_id, device_info[deviceInfo_id])
        
        # gst_hls(deviceInfo_id, device_info[deviceInfo_id])
        # push_status = device_hls_push(deviceInfo_id, device_info)
        # print(push_status)

    if msg.subject == "service.member_update":
        
        print(msg.data)   
        data = (msg.data)
        #print(data)
        data  = data.decode()
        data = json.loads(data)
        print(data)
        status = add_member_to_lmdb(data)
        if status:
            subject = msg.subject
            reply = msg.reply
            data = msg.data.decode()
            await nc_client.publish(msg.reply,b'ok')
            print("Received a message on '{subject} {reply}': {data}".format(
                subject=subject, reply=reply, data=data))
            load_lmdb_list()

def load_lmdb_fst(mem_data):
    i = 0
    for each in mem_data:
        i = i+1
        add_member_to_lmdb(each)
        print("inserting ",each)
        
# def load_lmdb_list_datainfo():
#     known_whitelist_faces1, known_whitelist_id1 = attendance_lmdb_known()
#     known_blacklist_faces1, known_blacklist_id1 = attendance_lmdb_unknown()
#     return [known_whitelist_faces1,known_blacklist_faces1,known_whitelist_id1, known_blacklist_id1]
        
# def get_info():
#     remove_cnts("./lmdb")
#     print("removed lmdb contents")
#     mem_data = fetch_db_mem()
#     # print(mem_data)
    
#     load_lmdb_fst(mem_data)
#     data_info = load_lmdb_list_datainfo()
#     return data_info

async def main():
    try:

        remove_cnts("./lmdb")
        load_lmdb_list()
        print("removed lmdb contents")
        mem_data = fetch_db_mem()
        print(mem_data)
        
        load_lmdb_fst(mem_data)
        load_lmdb_list()

        device_data = fetch_db()
        # # print(device_data)
        call_gstreamer(device_data)

        await nc_client.connect(servers=nats_urls) # Connect to NATS cluster!
        print("Nats Connected successfully!\n")
        await nc_client.subscribe("service.*", cb=device_info) # Subscribe to the device topic and fetch data through callback
        print("Subscribed to the topic, now you'll start receiving the Device details!\n")

    except Exception as e:
        await nc_client.close() # Close NATS connection
        print("Nats encountered an error: \n", e)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    loop = asyncio.get_event_loop()
    try :
        # asyncio.run(main())
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")
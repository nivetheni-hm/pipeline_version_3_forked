from io import BytesIO
import face_recognition 
import subprocess as sp
import cv2
from datetime import datetime  #datetime module to fetch current time when frame is detected
import numpy as np
from pytz import timezone 
from nanoid import generate
import random

face_did_encoding_store = dict()
TOLERANCE = 0.70
batch_person_id = []
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

known_whitelist_id = []
known_blacklist_id = []
unknown_faces = []
unknown_dids = []
unknown_id = []

person_did = ''

def find_person_type(im0, datainfo):
    
    global person_did
    
    known_whitelist_faces = datainfo[0]
    known_blacklist_faces = datainfo[1]
    known_whitelist_id_1 = datainfo[2]
    
    for x in known_whitelist_id_1:
        if x not in known_whitelist_id:
            known_whitelist_id.append(x)

    known_blacklist_id_1 = datainfo[3]
    
    for x in known_blacklist_id_1:
        if x not in known_blacklist_id:
            known_blacklist_id.append(x)

    def generate_unknown_dids(num_unknowns):
        prefix = "did:ckdr:Unknown"
        
        for i in range(num_unknowns):
            random_id = ''.join(random.choice('0123456789abcdef') for _ in range(16))
            unknown_dids.append(prefix + random_id)
        return unknown_dids

    # for i in unknown_faces:
    #     id_temp = generate_unknown_dids(1)
    #     if id_temp not in unknown_id:
    #         unknown_id.append(id_temp)
    combine_id = known_whitelist_id + known_blacklist_id + unknown_id
    
    print("WHITELIST ID SIZE: ", len(known_whitelist_id))
    print("BLACKLIST ID SIZE: ", len(known_blacklist_id))
    print("UNKNOWN ID SIZE: ", len(unknown_id))

    np_arg_src_list = known_whitelist_faces + known_blacklist_faces + unknown_faces
    # print(len(np_arg_src_list))
    np_bytes2 = BytesIO()
    np.save(np_bytes2, im0, allow_pickle=True)
    np_bytes2 = np_bytes2.getvalue()
    # print(combine_id)
    # print(known_whitelist_id)
    # print(known_blacklist_id)
    
    names = []
    for id_str in combine_id:
        print("ID STRING: ", id_str)
        name_start_index = id_str.find("Sat") + len("Sat")
        name = id_str[name_start_index:]
        # print(name)
        names.append(name)

    image = im0 # if im0 does not work, try with im1
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    locations = face_recognition.face_locations(image)
    # print("locations",locations)
    #encodings = face_recognition.face_encodings(image, locations, model = "large")
    encodings = face_recognition.face_encodings(image,locations,model = "large")
 
    did = "" 
    track_type = "100"
    if len(locations) != 0:
        if len(known_whitelist_faces) and len(known_blacklist_faces):
            for face_encoding ,face_location in zip(encodings, locations):
                results_combined = face_recognition.compare_faces(np_arg_src_list, face_encoding, TOLERANCE)
                faceids_combined = face_recognition.face_distance(np_arg_src_list,face_encoding)
                results_whitelist = face_recognition.compare_faces(known_whitelist_faces, face_encoding, TOLERANCE)
                faceids_whitelist = face_recognition.face_distance(known_whitelist_faces,face_encoding)
                results_blacklist = face_recognition.compare_faces(known_blacklist_faces, face_encoding, TOLERANCE)
                faceids_blacklist = face_recognition.face_distance(known_blacklist_faces,face_encoding)
                results_unknown = face_recognition.compare_faces(unknown_faces, face_encoding, TOLERANCE)
                faceids_unknown = face_recognition.face_distance(unknown_faces,face_encoding)
                # print(results_combined)
                # print(faceids_combined)
                # print(np.argmin(faceids_combined))
                matchindex_combined = np.argmin(faceids_combined)
                if faceids_combined[matchindex_combined] <=0.57:
                    print(names[matchindex_combined])
                    if results_combined[matchindex_combined]:
                        did = str(combine_id[matchindex_combined])
                        pattern = re.compile(r"Unknown")
                        if pattern.search(did):
                            print("UNKNOWN IDENTITY REPEATING:", did)
                        # batch_person_id.append(did)
                        else:
                            person_did = did
                            print("PERSON DETECTED: ",did)
                else:
                    print("UNKNOWN FACE DETECTED")
                    unknown_faces.append(face_encoding)
                 
    for i in unknown_faces:
        id_temp = generate_unknown_dids(1)
        if id_temp in unknown_id:
            continue
        else:    
            unknown_id.append(id_temp)
            person_did = str(id_temp)
            # batch_person_id.append(id_temp)
    # if track_type != "100":
    print("*****************************************")
    print("*****************************************")
    print("*****************************************")
    print("DID: ", did)
    if len(unknown_id) !=0:
        print(unknown_id)
    print("Unknown faces detected: ",len(unknown_faces))
    print("*****************************************")
    print("*****************************************")
    print("*****************************************")

    return person_did
                    
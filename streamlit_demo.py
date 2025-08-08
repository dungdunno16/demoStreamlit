import numpy
import cv2
import glob
import os
import re

import numpy as np
import torch
import paddle
from jupyter_core.version import pattern
from paddleocr import PaddleOCR
from ultralytics import YOLO
import streamlit as st

ocr = PaddleOCR(use_angle_cls=True,lang="en")
plate_code = [68,67,99, 98,69, 94,65, 83, 95,11,43, 92,47, 78,27,60, 39, 93,66, 63,
              81, 77,29, 30,31,32,33,40,38,15, 16, 34,89, 17,79, 85,25,49, 48, 86,12,
              24, 21,37,35, 18, 90,19, 28, 88,76, 82,14,74, 73,26,70, 62,20, 97,36,75,41,
              50, 51,52,53,54,55,56,57,58,59,61,72,22, 23,64, 71, 84,80]

def paddle_ocr(frame):
    # frame = frame[int(y1):int(y2),int(x1):int(x2)]
    result = ocr.ocr(frame,cls=False)
    plate = ""
    if result and result[0]:
        for line in result[0]:
            box,(text,score)=line
            if score > 0.8:
                plate += text
        plate = plate.strip()
        pattern = re.compile('[\W]')
        plate = pattern.sub('',plate)
        plate = plate.replace("???","")
    return plate

def check_plate(plate):
    if (
            (len(plate) > 5 and len(plate) < 10)
            and plate[2].isalpha()
            and all(char.isdigit() for i,char in enumerate(plate) if i != 2 and i != 3)
            and (plate[:2].isdigit() and int(plate[:2]) in plate_code)
    ):
        return True
    else:
        return False


st.header('Demo')

uploaded_file = st.file_uploader("Choose a file")

model = YOLO('yolov8s100epochplatev5(2)best.pt')
coco_model = YOLO('yolov8s.pt')

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    file_bytes = uploaded_file.read()
    np_array = np.frombuffer(file_bytes,np.uint8)
    img = cv2.imdecode(np_array,cv2.IMREAD_COLOR)
    # img = cv2.imread(upload_img)

    result = model.predict(img)
    for r in result:
        x1, y1, x2, y2 = r.boxes.xyxy[0]
        print(int(x1), int(y1), int(x2), int(y2))
    plate_img = img[int(y1):int(y2), int(x1):int(x2)]
    plate_text = paddle_ocr(plate_img)
    st.image(plate_img,caption=plate_text,use_column_width=True)
    print(plate_text)

# url = 'http://192.168.1.205:8080/video'
# # video = cv2.VideoCapture(0)
# video = cv2.VideoCapture('test.MOV')
# # video = cv2.VideoCapture(url)
#
# fps = video.get(cv2.CAP_PROP_FPS)
# width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
#
# # video.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
# vehicles =[2,3,5]
# ret = True
# while ret:
#     ret,frame = video.read()
#     if ret:
#         detection = coco_model.track(frame, persist=True)[0]
#         if detection.boxes is not None:
#             for det in detection.boxes.data.tolist():
#                 if len(det) == 7:
#                     x1, y1, x2, y2, track_id, score, class_id = det
#                 if int(class_id) in vehicles and score > 0.5:
#                     vehicles_bbox = []
#                     vehicles_bbox.append([x1,y1,x2,y2,track_id,score])
#                     for bbox in vehicles_bbox:
#                         frame_vehicle = frame[int(y1):int(y2),int(x1):int(x2)]
#                         license_plates = model(frame_vehicle)[0]
#                         if license_plates.boxes is not None:
#                             for license_plate in license_plates.boxes.data.tolist():
#                                 plate_x1, plate_y1, plate_x2, plate_y2, plate_score,_ = license_plate
#                                 plate = frame_vehicle[int(plate_y1):int(plate_y2),int(plate_x1):int(plate_x2)]
#                                 plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
#                                 _, plate_threshold = cv2.threshold(plate_gray,64,255,cv2.THRESH_BINARY_INV)
#                                 plate_text = paddle_ocr(plate_threshold)
#                                 if len(plate_text) > 0:
#                                     cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)),(241,245,246),2)
#                                     if check_plate(plate_text):
#                                         cv2.rectangle(frame_vehicle, (int(plate_x1),int(plate_y1)),(int(plate_x2),int(plate_y2)), (235, 77, 75),2)
#                                         cv2.putText(frame_vehicle,plate_text,(int(plate_x1),int(plate_y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(235,77,75),2)
#         out.write(frame)
# video.release()
# out.release()
# cv2.destroyAllWindows()

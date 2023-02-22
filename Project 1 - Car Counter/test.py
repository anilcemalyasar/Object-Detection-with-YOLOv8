from math import ceil
from ultralytics import YOLO
import cv2
import cvzone
from sort import *
import numpy as np

model = YOLO("../Yolo-Weights/yolov8l.pt")

cap = cv2.VideoCapture("../Videos/cars.mp4")
cap.set(3, 1280)
cap.set(4, 720)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# Line Coordinates
limits = [400, 297, 673, 297]
totalCount = []

while True:

    success, img = cap.read()
    imgReg = cv2.bitwise_and(img, mask)
    results = model(source=imgReg, stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            #Confidence
            conf = ceil(box.conf[0] * 100) / 100
            #Class Names
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                #cvzone.cornerRect(img, (x1,y1,w,h), l=9)
                #cvzone.putTextRect(img, f"{currentClass} {conf}", (max(0,x1), max(35,y1)), scale = 0.8, thickness=1, offset=1)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    # Line
    cv2.line(img, (limits[0], limits[1]), (limits[2],limits[3]), (255,0,255), thickness=3)

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR = (255,0,255))
        cvzone.putTextRect(img, f"{int(id)} {conf}", (max(0,x1), max(35,y1)), scale=1, thickness=2, offset=5)

        # Center of detected objects
        cx, cy = x1 + w//2, y1 + h//2
        cv2.circle(img, (cx,cy), radius=5, color=(0,255,0), thickness=cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0,255,0), thickness=3)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
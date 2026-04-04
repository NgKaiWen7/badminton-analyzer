from ultralytics import YOLO
import torch
import cv2

model = YOLO("yolov8n-pose.pt")

image = cv2.imread("test.png")
img = model.preprocess(image)  # internal preprocessing

# Forward pass through the raw model
with torch.no_grad():
    preds = model.model(img)   # <-- raw output

print(type(preds))
print(preds[0].shape)
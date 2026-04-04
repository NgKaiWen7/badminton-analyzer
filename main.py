# Install ultralytics if not yet installed:
# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

# Load YOLOv8 pose model (pretrained)
model = YOLO("yolov8n-pose.pt")  # lightweight pose model, you can change to yolov8m-pose.pt for more accuracy

# Load an image
image_path = "test.png"  # path to your image
image = cv2.imread(image_path)

# Run pose estimation
results = model.predict(image)
print(results[0].boxes.data)
print(results[0].boxes.conf)
exit()

for result in results:
    xy = result.keypoints.xy
    xyn = result.keypoints.xyn
    kpts = result.keypoints.data
    print(kpts)

    xy = xy.numpy()
    for x, y in xy[0]:
        cv2.circle(image, [int(x), int(y)], 3, (0, 0, 255), -1)
    cv2.imwrite("output.jpg", image)
from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26n-pose.pt")

model.ir_version = 9
# Export the model to ONNX format
model.export(
    format="onnx",
    simplify=True,
    opset=12,
    dynamic=False
)
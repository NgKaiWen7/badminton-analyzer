g++ -g main.cpp \
model/model.cpp \
utils/nms.cpp \
drawing/drawing.cpp \
pipeline/process.cpp \
-o yolo \
-I. \
-I/opt/onnxruntime/include \
-L/opt/onnxruntime/lib \
-lonnxruntime \
-Wl,-rpath,/opt/onnxruntime/lib \
`pkg-config --cflags --libs opencv4`
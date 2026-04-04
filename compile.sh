g++ -g main.cpp \
inference/inference.cpp \
utils/math.cpp \
postprocess/nms.cpp \
postprocess/decode.cpp \
postprocess/drawing.cpp \
-o yolo \
-I. \
-I/opt/onnxruntime/include \
-L/opt/onnxruntime/lib \
-lonnxruntime \
-Wl,-rpath,/opt/onnxruntime/lib \
`pkg-config --cflags --libs opencv4`
# g++ -g \
# main.cpp \
# model/model.cpp \
# utils/nms.cpp \
# drawing/drawing.cpp \
# pipeline/process.cpp \
# model/kalmanfilter.cpp \
# -o yolo \
# -I. \
# -I./eigen \
# -I./detection \
# -I./model \
# -I./utils \
# -I./drawing \
# -I./pipeline \
# -I/opt/onnxruntime/include \
# -L/opt/onnxruntime/lib \
# -lonnxruntime \
# -Wl,-rpath,/opt/onnxruntime/lib \
# `pkg-config --cflags --libs opencv4`
# 
# exit

g++ -std=c++17 -fPIC -shared \
main.cpp \
model/model.cpp \
utils/nms.cpp \
drawing/drawing.cpp \
pipeline/process.cpp \
-o libinfer.so \
-I. \
-I./eigen \
-I./detection \
-I./model \
-I./utils \
-I./drawing \
-I./pipeline \
-I/opt/onnxruntime/include \
-L/opt/onnxruntime/lib \
-lonnxruntime \
-I/opt/bytetrack/include \
-Wl,-rpath,/opt/onnxruntime/lib \
`pkg-config --cflags --libs opencv4`
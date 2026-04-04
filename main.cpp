#include "inference/inference.h"
#include "postprocess/decode.h"
#include "postprocess/drawing.h"
#include <iostream>

int main()
{
    Inference model("yolo26n-pose.onnx");

    cv::Mat img = cv::imread("test.png");
    int heights = img.rows;
    int width = img.cols;

    Ort::Value raw_output = model.run(img);
    float x_scale = width / 640.0;
    float y_scale = heights / 640.0;
    auto detections = decode_output(raw_output, x_scale, y_scale, 0.5);
    auto output = apply_nms(detections, 0.5);
    cv::Mat output_img;
    output_img = img.clone();
    draw_output(output, output_img);
    cv::imwrite("output.jpg", output_img);
    return 0;
}
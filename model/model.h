#pragma once
#include <vector>
#include <string>
#include <detection/detection.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class Model {
public:
    Model(const std::string& model_path);

    // Public interface: process a frame and draw output
    std::vector<Detection> process_frame(float conf, const cv::Mat& frame);

    int get_input_width() const { return model_input_width; }
    int get_input_height() const { return model_input_height; }

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    int model_input_width;
    int model_input_height;

    // Internal helpers
    Ort::Value run(const cv::Mat& image);
    std::vector<Detection> decode_output(Ort::Value& output, float x_scale, float y_scale, float conf);
};
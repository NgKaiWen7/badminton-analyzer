#pragma once
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class Inference {
public:
    Inference(const std::string& model_path);

    Ort::Value run(const cv::Mat& image);

private:
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    std::vector<std::string> input_names_str;
    std::vector<std::string> output_names_str;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    int input_width;
    int input_height;
};
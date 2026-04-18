#pragma once
#include <opencv2/opencv.hpp>
#include <detection/detection.h>

void draw_output(std::vector<Detection>& results, cv::Mat& output_image);
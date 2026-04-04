#pragma once
#include "postprocess/nms.h"
#include <opencv2/opencv.hpp>

void draw_output(std::vector<Detection>& results, cv::Mat& output_image);
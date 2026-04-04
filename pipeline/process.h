#pragma once
#include <opencv2/opencv.hpp>

class Model; // forward declaration

void process_image(const std::string& path, Model& model);
void process_video(const std::string& input, const std::string& output, Model& model);
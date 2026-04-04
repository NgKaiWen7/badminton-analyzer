#pragma once
#include <vector>

struct Detection {
    int index;
    float conf;
    std::vector<float> bounding_box;
    std::vector<float> key_points;
};

std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float iou_threshold
);
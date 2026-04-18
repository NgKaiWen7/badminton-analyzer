#pragma once
#include <vector>
#include "detection/detection.h"

std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float iou_threshold
);
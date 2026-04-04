#pragma once
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "nms.h"

std::vector<Detection> decode_output(
    Ort::Value& output,
    float x_scale,
    float y_scale,
    float conf_threshold
);
#include "nms.h"
#include "detection/detection.h"
#include <algorithm>
#include <iostream>

static float compute_iou(const Detection &a, const Detection &b)
{
    float ax = a.bounding_box.x;
    float ay = a.bounding_box.y;
    float aw = a.bounding_box.w;
    float ah = a.bounding_box.h;

    float bx = b.bounding_box.x;
    float by = b.bounding_box.y;
    float bw = b.bounding_box.w;
    float bh = b.bounding_box.h;

    float x1 = std::max(ax - aw / 2, bx - bw / 2);
    float y1 = std::max(ay - ah / 2, by - bh / 2);
    float x2 = std::min(ax + aw / 2, bx + bw / 2);
    float y2 = std::min(ay + ah / 2, by + bh / 2);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = aw * ah;
    float areaB = bw * bh;
    return inter / (areaA + areaB - inter);
}

std::vector<Detection> apply_nms(
    const std::vector<Detection>& detections,
    float iou_threshold)
{
    std::vector<Detection> result;

    auto sorted = detections;
    std::sort(sorted.begin(), sorted.end(),
              [](const Detection& a, const Detection& b)
              {
                  return a.bounding_box.conf > b.bounding_box.conf;
              });

    for (const auto& det : sorted)
    {
        bool keep = true;

        for (const auto& r : result)
        {
            float iou =  compute_iou(det, r);
            if (iou > iou_threshold)
            {
                keep = false;
                break;
            }
        }

        if (keep)
            result.push_back(det);
    }

    return result;
}
#pragma once
#include <vector>

struct Keypoints {
    int x;
    int y;
    float conf;
};

struct BoundingBox{
    int x;
    int y;
    int w;
    int h;
    float conf;
};

struct Detection {
    int index;
    BoundingBox bounding_box;
    std::vector<Keypoints> key_points;
};
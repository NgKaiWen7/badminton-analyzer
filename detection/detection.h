#pragma once
#include <vector>
#include <Eigen/Dense>

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
    Eigen::Vector4f to_tlwh() const {
        return Eigen::Vector4f(x, y, w, h);
    }

    Eigen::Vector4f to_xyxy() const {
        return Eigen::Vector4f(x, y, x + w, y + h);
    }

    Eigen::Vector2f center() const {
        return Eigen::Vector2f(x + w * 0.5f, y + h * 0.5f);
    }
};

struct Detection {
    int id;
    BoundingBox bounding_box;
    std::vector<Keypoints> key_points;
};
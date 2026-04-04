#include <opencv2/opencv.hpp>
#include "nms.h"
#include <vector>
void draw_output(std::vector<Detection>& results, cv::Mat& output_image)
{
    for (Detection& dect : results)
    {
        for (size_t i = 0; i < dect.key_points.size(); i += 3)
        {
            float x = dect.key_points[i];
            float y = dect.key_points[i + 1];
            float conf = dect.key_points[i + 2];

            cv::circle(output_image,
                        cv::Point(static_cast<int>(x), static_cast<int>(y)),
                        3,
                        cv::Scalar(0, 255, 0),
                        -1);
        }
    }
}
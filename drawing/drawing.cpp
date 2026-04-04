#include <opencv2/opencv.hpp>
#include <detection/detection.h>
#include <vector>

void draw_output(std::vector<Detection> &results, cv::Mat &output_image)
{
    for (Detection &dect : results)
    {
        for (Keypoints &keypoint : dect.key_points)
        {
            cv::Point point(keypoint.x, keypoint.y);
            cv::circle(output_image, point, 3, cv::Scalar(0, 255, 0), -1);
        }
    }
}
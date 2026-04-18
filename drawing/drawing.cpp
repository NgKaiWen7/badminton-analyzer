#include <opencv2/opencv.hpp>
#include <detection/detection.h>
#include <vector>

void draw_output(std::vector<Detection> &results, cv::Mat &output_image)
{
    for (Detection &dect : results)
    {
        cv::Rect box(
            static_cast<int>(dect.bounding_box.x - dect.bounding_box.w / 2),
            static_cast<int>(dect.bounding_box.y - dect.bounding_box.h / 2),
            static_cast<int>(dect.bounding_box.x + dect.bounding_box.w / 2),
            static_cast<int>(dect.bounding_box.y + dect.bounding_box.h / 2)
        );
        cv::rectangle(output_image, box, cv::Scalar(255, 0, 0), 2);
        for (Keypoints &keypoint : dect.key_points)
        {
            cv::Point point(keypoint.x, keypoint.y);
            cv::circle(output_image, point, 3, cv::Scalar(0, 255, 0), -1);
        }
    }
}
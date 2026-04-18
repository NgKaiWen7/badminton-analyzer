#include <opencv2/opencv.hpp>
#include <detection/detection.h>
#include <vector>

void draw_output(std::vector<Detection> &results, cv::Mat &output_image)
{
    for (Detection &dect : results)
    {
        cv::Rect box(
            static_cast<int>(dect.bounding_box.x),
            static_cast<int>(dect.bounding_box.y),
            static_cast<int>(dect.bounding_box.w-dect.bounding_box.x),
            static_cast<int>(dect.bounding_box.h-dect.bounding_box.y)
        );
        cv::rectangle(output_image, box, cv::Scalar(255, 0, 0), 2);
        std::string label = "ID: " + std::to_string(dect.id);
        cv::putText(output_image, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        for (Keypoints &keypoint : dect.key_points)
        {
            cv::Point point(keypoint.x, keypoint.y);
            cv::circle(output_image, point, 3, cv::Scalar(0, 255, 0), -1);
        }
    }
}
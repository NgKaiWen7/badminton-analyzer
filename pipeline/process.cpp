#include <iostream>
#include <opencv2/opencv.hpp>
#include <drawing/drawing.h>
#include <detection/detection.h>
#include <model/model.h>
#include "bytetrack/bytetrack.hpp"

void process_video(const std::string& input,
                   const std::string& output,
                   Model& model)
{
    cv::VideoCapture cap(input);

    int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(
        output,
        cv::VideoWriter::fourcc('m','p','4','v'),
        fps,
        cv::Size(w, h)
    );

    cv::Mat frame;

    int processed_frame = 0;
    bytetrack::ByteTracker tracker(0.8f, 0.2f, 0.3f, 25);
    while (cap.read(frame)) {
        processed_frame ++;
        std::cout << "Processing Frame: " << processed_frame << std::endl;
        std::vector<Detection> detection = model.process_frame(0.2, frame);
        std::vector<bytetrack::DetectionBox> results;

        for (const auto& det : detection) {
            const auto& box = det.bounding_box;
        
            float x1 = static_cast<float>(box.x);
            float y1 = static_cast<float>(box.y);
            float x2 = static_cast<float>(box.w);
            float y2 = static_cast<float>(box.h);
            float conf = box.conf;
            int cls = 0;
            results.push_back({x1, y1, x2, y2, conf, cls});
        }
        
        std::vector<bytetrack::TrackResult> tracked_output = tracker.update(results);
        
        std::vector<Detection> tracked_detections;
        for (auto& track : tracked_output) {
            Detection& det = detection[track.idx];
            det.id = track.track_id;
            tracked_detections.push_back(det);
        }
        draw_output(tracked_detections, frame);
        writer.write(frame);
    }
}

void process_image(const std::string& path, Model& model)
{
    cv::Mat img = cv::imread(path);
    std::vector<Detection> detection = model.process_frame(0.5, img);
    draw_output(detection, img);
    cv::imwrite("output.jpg", img);
}
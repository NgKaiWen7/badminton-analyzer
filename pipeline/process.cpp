#include <iostream>
#include <opencv2/opencv.hpp>
#include <drawing/drawing.h>
#include <detection/detection.h>
#include <model/model.h>

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
    while (cap.read(frame)) {
        processed_frame ++;
        std::cout << "Processing " << processed_frame << std::endl;
        std::vector<Detection> detection = model.process_frame(0.2, frame);
        draw_output(detection, frame);
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
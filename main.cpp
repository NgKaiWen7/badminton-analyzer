#include <iostream>
#include <string>
#include "pipeline/process.h"
#include "model/model.h"
#include "bytetrack/bytetrack.hpp"

bool is_image(const std::string& path)
{
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return (ext == "jpg" || ext == "jpeg" || ext == "png");
}

bool is_video(const std::string& path)
{
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return (ext == "mp4" || ext == "h265" || ext == "avi" || ext == "mkv");
}

extern "C" {

void run_inference(const char* model_path,
                   const char* input_path,
                   const char* output_path)
{
    Model model(model_path);

    std::string input(input_path);
    std::string output(output_path);

    if (is_image(input))
    {
        process_image(input, model);
    }
    else if (is_video(input))
    {
        process_video(input, output, model);
    }
}

}

/*
int main() {
    KalmanFilter kf;

    Eigen::Matrix<float, 4, 1> z;
    z << 100, 200, 0.5, 180;

    kf.initiate(z);

    kf.predict();
    kf.set_noise(kf.x(3));
    kf.update(z);
    std::cout<<"Success" << std::endl;

    return 0;
}*/
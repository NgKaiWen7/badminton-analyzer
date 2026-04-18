#include "detection.h"
#include "model.h"
#include "utils/nms.h"
#include <iostream>
#include <vector>

Model::Model(const std::string &model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      memory_info(Ort::MemoryInfo::CreateCpu(
          OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0; // GPU 0
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = SIZE_MAX;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.do_copy_in_default_stream = 1;
    session = Ort::Session(env, model_path.c_str(), session_options);

    // Get input info
    auto input_name = session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    input_names_str.push_back(input_name.get());
    input_names.push_back(input_names_str.back().c_str());

    auto input_shape = session.GetInputTypeInfo(0)
                           .GetTensorTypeAndShapeInfo()
                           .GetShape();

    model_input_height = input_shape[2];
    model_input_width = input_shape[3];

    // Get output info
    auto output_name = session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    output_names_str.push_back(output_name.get());
    output_names.push_back(output_names_str.back().c_str());
    std::cout << output_names_str[0] << std::endl;
}

Ort::Value Model::run(const cv::Mat &image)
{
    // 1. Resize
    cv::Mat resized;
    int input_width = get_input_width();
    int input_height = get_input_height();
    cv::resize(image, resized, cv::Size(input_width, input_height));

    // 2. Convert to float
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // 3. HWC → CHW
    std::vector<float> input_tensor_values;
    input_tensor_values.resize(3 * input_width * input_height);
    std::vector<cv::Mat> chw(3);
    for (int i = 0; i < 3; ++i)
    {
        chw[i] = cv::Mat(input_height, input_width, CV_32F, input_tensor_values.data() + i * input_width * input_height);
    }
    cv::split(resized, chw);

    // 4. Create tensor
    std::vector<int64_t> input_shape = {1, 3, input_height, input_width};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    // 5. Run model
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1);

    return std::move(output_tensors[0]);
}

std::vector<Detection> Model::decode_output(Ort::Value &output, float x_scale, float y_scale, float conf_threshold)
{
    auto shape_info = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();

    int num_preds = shape[1];
    int num_features = shape[2];

    std::vector<Detection> valid_predictions;
    float *data = output.GetTensorMutableData<float>();
    for (int i = 0; i < num_preds; ++i)
    {
        const float *pred = data + i * num_features;
        float obj_conf = pred[4];
        if (obj_conf >= conf_threshold)
        {
            BoundingBox bounding_box;
            bounding_box.x = int(pred[0] * x_scale);
            bounding_box.y = int(pred[1] * y_scale);
            bounding_box.w = int(pred[2] * x_scale);
            bounding_box.h = int(pred[3] * y_scale);
            bounding_box.conf = obj_conf;
            std::vector<Keypoints> keypoints;
            for (int k = 6; k < 57; k += 3)
            {
                Keypoints keypoint;
                keypoint.x = int (pred[k] * x_scale);
                keypoint.y = int (pred[k+1] * y_scale);
                keypoint.conf = pred[k+2];
                keypoints.push_back(keypoint);
            }
            Detection dect = {i, bounding_box, keypoints};
            valid_predictions.push_back(dect);
        }
    }
    return valid_predictions;
}

std::vector<Detection> Model::process_frame(float conf, const cv::Mat &frame)
{
    float h = frame.rows;
    float w = frame.cols;

    auto raw_output = run(frame);

    float x_scale = w / get_input_width();
    float y_scale = h / get_input_height();

    auto detections = decode_output(raw_output, x_scale, y_scale, conf);
    // auto output = apply_nms(detections, 0.8);
    return detections;
}
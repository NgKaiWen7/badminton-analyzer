#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

float compute_iou(const Detection &a, const Detection &b)
{
    float x1 = std::max(a.x - a.w / 2, b.x - b.w / 2);
    float y1 = std::max(a.y - a.h / 2, b.y - b.h / 2);
    float x2 = std::min(a.x + a.w / 2, b.x + b.w / 2);
    float y2 = std::min(a.y + a.h / 2, b.y + b.h / 2);

    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    float area_a = a.w * a.h;
    float area_b = b.w * b.h;

    return inter_area / (area_a + area_b - inter_area);
}

int search_for_best_output(Ort::Value &output, float conf_threshold)
{
    // Get tensor info
    Ort::TensorTypeAndShapeInfo shape_info = output.GetTensorTypeAndShapeInfo();

    // Get shape
    std::vector<int64_t> shape = shape_info.GetShape();

    // Get data
    float *data = output.GetTensorMutableData<float>();

    int num_preds = shape[1];
    int num_features = shape[2];

    int best_prediction = -1;
    float best_confidence = 0;

    for (int i = 0; i < num_preds; ++i)
    {
        float obj_conf = data[num_features * i];
        obj_conf = sigmoid(obj_conf);
        if (obj_conf > 1)
        {
            std::cerr << "Wrong obj conf" << std::endl;
        }
        else if (obj_conf > conf_threshold && obj_conf > best_confidence)
        {
            best_prediction = i;
            best_confidence = obj_conf;
            std::cout << obj_conf << std::endl;
        }
    }
    return best_prediction;
}

void prepare_input(const cv::Mat &original_input, std::vector<float> &input_data)
{
    int channels = 3;
    int height = 640;
    int width = 640;
    std::vector<cv::Mat> chw(3);

    cv::Size new_size(width, height);
    cv::Mat resized_input;
    cv::resize(original_input, resized_input, new_size);
    resized_input.convertTo(resized_input, CV_32F, 1.0 / 255.0);

    for (int i = 0; i < 3; ++i){
        chw[i] = cv::Mat(height, width, CV_32F, input_data.data() + i * width * height);
    }
    cv::split(resized_input, chw);
}

int main()
{
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;  // GPU 0
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = SIZE_MAX;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
    cuda_options.do_copy_in_default_stream = 1;

    session_options.AppendExecutionProvider_CUDA(cuda_options);
    Ort::Session session(env, "yolo26n-pose.onnx", session_options);
    std::cout << "Model loaded\n";

    cv::VideoCapture cap("sample.mp4");
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video\n";
        return -1;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Size original_size(width, height);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::array<int64_t, 4> input_shape{1, 3, 640, 640};
    std::vector<float> input_data(1 * 3 * 640 * 640);
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

    const char *input_name = input_name_alloc.get();
    const char *output_name = output_name_alloc.get();

    cv::VideoWriter writer(
        "output.mp4",
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        cv::Size(640, 640));

    cv::Mat frame;
    int frame_count = 0;
    int max_frame = 20 * fps;
    while (cap.read(frame) && frame_count < max_frame)
    {
        cv::Mat output_frame;
        cv::resize(frame, output_frame, cv::Size(640, 640));
        prepare_input(frame, input_data);

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor,
            1,
            &output_name,
            1);

        Ort::Value &output = output_tensors[0];
        int best_output = search_for_best_output(output, 0.5);
        if (best_output >= 0){
            output_inference_result(output, best_output, output_frame);
            // cv::resize(output_frame, output_frame, original_size);
        }
        writer.write(output_frame);
        frame_count++;
    }
    return 0;
}

#include "decode.h"
#include "nms.h"
#include "utils/math.h"
#include <iostream>

std::vector<Detection> decode_output(Ort::Value &output, float x_scale, float y_scale, float conf_threshold)
{
    auto shape_info = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();

    int num_preds = shape[1];
    int num_features = shape[2];
    std::cout << num_features << num_preds << std::endl;

    std::vector<Detection> valid_predictions;
    float *data = output.GetTensorMutableData<float>();
    for (int i = 0; i < num_preds; ++i)
    {
        const float* pred = data + i * num_features;
        float obj_conf = pred[4];
        if (obj_conf >= conf_threshold)
        {
            std::vector<float> bounding_box = {pred[0] * x_scale, pred[1] * y_scale, pred[2] * x_scale, pred[3] * y_scale};
            std::vector<float> keypoints;
            for (int k = 6; k < 57; k += 3) {
                float kx = pred[k] * x_scale;
                float ky = pred[k+1] * y_scale;
                float kc = pred[k+2];
                //std::cout << y_scale << std::endl;
                //std::cout << x_scale << std::endl;
                keypoints.push_back(kx);
                keypoints.push_back(ky);
                keypoints.push_back(kc);
            }
            Detection dect = {i, obj_conf, bounding_box, keypoints};
            valid_predictions.push_back(dect);
        }
    }
    return valid_predictions;
}
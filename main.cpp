#include <iostream>
#include "model/model.h"
#include "pipeline/process.h"

int main()
{
    Model model("yolo26n-pose.onnx");
    process_image("test.png", model);

    return 0;
}
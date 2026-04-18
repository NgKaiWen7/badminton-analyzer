FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    pkg-config \
    gdb \
    libcudnn9-cuda-13 \
    python3 \
    python3-pip

# install onnxruntime
RUN mkdir -p /opt/onnxruntime \
    && wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-gpu_cuda13-1.24.4.tgz \
    && tar -xvf onnxruntime-linux-x64-gpu_cuda13-1.24.4.tgz --strip-components=1 -C /opt/onnxruntime
    # && mv onnxruntime-linux-x64-gpu /opt/onnxruntime 

ENV ONNXRUNTIME_DIR=/opt/onnxruntime
COPY . /workspace
COPY bytetrack /opt/bytetrack
WORKDIR /workspace
RUN pip install --no-cache-dir python-multipart fastapi uvicorn numpyi
RUN bash compile.sh
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
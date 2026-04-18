import os
import uuid
import ctypes
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

# -------------------------
# Config
# -------------------------
MODEL_PATH = "model.onnx"
LIB_PATH = "./libinfer.so"
TMP_DIR = "/tmp"

os.makedirs(TMP_DIR, exist_ok=True)

# -------------------------
# Load C++ shared library
# -------------------------
lib = ctypes.CDLL(LIB_PATH)

# define function signatures
lib.run_inference.argtypes = [
    ctypes.c_char_p,  # model
    ctypes.c_char_p,  # input
    ctypes.c_char_p   # output
]

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # generate unique filenames
    ext = file.filename.split(".")[-1]
    input_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.{ext}")
    output_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}_out.mp4")

    # save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # call C++ inference
    lib.run_inference(
        b"yolo26n-pose.onnx",
        input_path.encode(),
        output_path.encode()
    )

    # return result file
    if os.path.exists(output_path):
        return FileResponse(output_path, media_type="video/mp4")

    # fallback (e.g. image case)
    return {"message": "processed", "input": input_path}
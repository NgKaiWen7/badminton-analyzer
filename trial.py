import requests

url = "http://127.0.0.1:8000/infer"

with open("/home/nkw/Downloads/test1.mp4", "rb") as f:
    files = {"file": ("/home/nkw/Downloads/test1.mp4", f, "video/mp4")}
    response = requests.post(url, files=files)

print(response.status_code)

# if video returned
with open("output.mp4", "wb") as out:
    out.write(response.content)
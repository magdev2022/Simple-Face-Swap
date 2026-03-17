# FaceSwap (ONNX)

A simple real-time face swapping and enhancement **Python** demo using ONNX models and `insightface`.

This project uses a webcam feed to swap your face with a source face image, then performs face restoration with CodeFormer.

---

## ✅ Features

- Face detection + landmark tracking with **InsightFace**
- Face swapping with **inswapper_128_fp16.onnx**
- Face enhancement with **codeformer**
---

## 🧩 Required Models (already included)

Place these files in the `models/` folder (already present in this repository):

- `models/det_10g.onnx` (detector)
- `models/w600k_r50.onnx` (recognition)
- `models/inswapper_128_fp16.onnx` (face swap)
- `models/faceparser_fp16.onnx` (segmentation)
- `models/codeformer.onnx` (face enhance)
---

## 🛠 Setup (Windows)

1. **Activate the virtual environment** (already created in this repo):

    ```powershell
    & .\.venv\Scripts\Activate.ps1
    ```

2. **Install dependencies** (if not already installed):

    ```powershell
    pip install -r requirements.txt
    ```
---

## ▶️ Run

From the repo root:

```powershell
python main.py
```

## 📌 Notes

- If you get slow performance, ensure `onnxruntime-gpu` is installed and your GPU is supported.
- The script assumes a single face in the source image; if multiple faces exist, only the first detected face is used.

---

## 🧪 Troubleshooting

- **No camera stream**: check your webcam is not in use by another app.
- **Model load failures**: ensure the `models/` folder contains the correct ONNX files and paths are unchanged.
- **Black/blank output**: verify OpenCV can open display windows (non-headless environment).

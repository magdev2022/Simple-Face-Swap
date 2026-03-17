import cv2
import numpy as np
import onnxruntime as ort
import os
import time
import traceback
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


# Model Paths
DET_MODEL = "models/det_10g.onnx"
SWAP_MODEL = "models/inswapper_128_fp16.onnx"
CODEFORMER_MODEL = "models/codeformer.onnx"
PARSER_MODEL = "models/faceparser_fp16.onnx"

providers = ['CUDAExecutionProvider','CPUExecutionProvider']


app = FaceAnalysis(
    name="buffalo_l",
    providers=providers
)

app.prepare(ctx_id=0, det_size=(640,640))


# Swap Model
swapper = get_model(SWAP_MODEL, providers=providers)


# Load ONNX Sessions for Enhancement
faceparser = ort.InferenceSession(PARSER_MODEL, providers=providers)
codeformer = ort.InferenceSession(CODEFORMER_MODEL, providers=providers)


# Helper Functions


def preprocess(img, size=(512,512)):
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    img = img.transpose(2,0,1)
    img = np.expand_dims(img,0)
    return img

def postprocess(out):
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.ndim == 4 and out.shape[0] == 1:
        out = out[0]
    out = out.transpose(1, 2, 0)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


# Face Segmentation
def run_faceparser(face):
    inp = preprocess(face,(512,512))
    name = faceparser.get_inputs()[0].name
    out = faceparser.run(None,{name:inp})
    mask = out[0][0].argmax(0).astype(np.uint8)
    mask = cv2.resize(mask,(face.shape[1],face.shape[0]))
    mask = (mask>0).astype(np.uint8)*255
    return mask

# CodeFormer Enhancement
def run_codeformer(face, weight=0.6):
    inp = preprocess(face,(512,512))
    input_name = codeformer.get_inputs()[0].name
    weight_name = codeformer.get_inputs()[1].name
    weight_val = np.array(weight, dtype=np.float64)
    out = codeformer.run(None, {input_name: inp, weight_name: weight_val})
    return postprocess(out)



# Load Source Face
source = cv2.imread("Taylor swift.jpg")
src_faces = app.get(source)

if len(src_faces)==0:
    print("No face in source image")
    exit()

source_face = src_faces[0]


# Webcam
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # compute FPS
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    faces = app.get(frame)

    result = frame.copy()

    for face in faces:

        # Face Swap
        result = swapper.get(
            result,
            face,
            source_face,
            paste_back=True
        )

        x1,y1,x2,y2 = map(int,face.bbox)

        face_crop = result[y1:y2,x1:x2]

        if face_crop.size == 0:
            continue

        mask = run_faceparser(face_crop)

        # Enhancement with CodeFormer
        face_restored = run_codeformer(face_crop)

        # Resize to match the region
        region = result[y1:y2, x1:x2]
        h_box, w_box = region.shape[:2]
        face_restored = cv2.resize(face_restored, (w_box, h_box))
        if face_restored.shape[0] != h_box or face_restored.shape[1] != w_box:
            face_restored = face_restored[:h_box, :w_box]

        # Blend using the face mask to avoid rectangular appearance
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_normalized] * 3, axis=-1)
        original_region = region.astype(np.float32)
        enhanced_region = face_restored.astype(np.float32)
        blended = original_region * (1 - mask_3d) + enhanced_region * mask_3d
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

    # show FPS on frame
    cv2.putText(result, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    try:
        cv2.imshow("Enhanced Face Swap", result)
    except cv2.error:      
        pass

    try:
        if cv2.waitKey(1) == 27:
            break
    except cv2.error:        
        break

cap.release()
cv2.destroyAllWindows()

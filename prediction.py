import os
import time
import cv2
from ultralytics import YOLO
from threading import Lock
_infer_lock = Lock()

MODEL_M1_PATH = "models/best_m1_v2.pt"
MODEL_M2_PATH = "models/best_m2_v1.pt"

model_m1 = YOLO(MODEL_M1_PATH)
model_m2 = YOLO(MODEL_M2_PATH)

# filesystem output locations (relative to project root)
OUT_M1_FS = os.path.join("static", "uploads", "m1", "output_image_m1.jpg")
OUT_M2_FS = os.path.join("static", "uploads", "m2", "output_image_m2.jpg")

# URL paths used by the browser
OUT_M1_URL = "/static/uploads/m1/output_image_m1.jpg"
OUT_M2_URL = "/static/uploads/m2/output_image_m2.jpg"


def _predict(model: YOLO, image_file: str, out_fs: str, out_url: str) -> str:
    with _infer_lock:
        start_time = time.time()
        results = model(image_file, conf=0.5, iou=0.6, imgsz=640)
        inference_time = time.time() - start_time

        for result in results:
            im_bgr = result.plot().copy()  # BGR

            # Build text lines
            lines = [f"Prediction Speed: {inference_time:.2f} s"]
            if result.boxes is not None and len(result.boxes) > 0:
                for cls_id, conf in zip(result.boxes.cls, result.boxes.conf):
                    lines.append(f"{result.names[int(cls_id)]}: {float(conf):.2f}")

            h, w = im_bgr.shape[:2]
            font_scale = max(w / 1200.0, 0.5)
            thickness = max(int(w / 600.0), 1)

            x, y0 = 10, max(h - 20 * len(lines), 30)
            for i, line in enumerate(lines):
                y = y0 + i * 20
                cv2.putText(im_bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

            os.makedirs(os.path.dirname(out_fs), exist_ok=True)
            ok = cv2.imwrite(out_fs, im_bgr)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for path: {out_fs}")

            return out_url

    return out_url


def get_prediction_m1(image_file: str) -> str:
    return _predict(model_m1, image_file, OUT_M1_FS, OUT_M1_URL)


def get_prediction_m2(image_file: str) -> str:
    return _predict(model_m2, image_file, OUT_M2_FS, OUT_M2_URL)

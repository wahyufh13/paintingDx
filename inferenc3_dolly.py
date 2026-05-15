import cv2
import time
import requests
import base64
import numpy as np
import re
import threading
from collections import Counter
from paddleocr import PaddleOCR

# =========================================================
# CONFIG
# =========================================================
CAM_INDEX = 0
OUTPUT_PATH = "output.mp4"

OCR_INTERVAL = 5
HIST_MAX = 10

OCR_LANG = "en"
USE_GPU = False
MIN_CONF = 0.5

FLIP_HORIZONTAL = False

ENABLE_API = False
ORDS_URL = "http://idtbintranetdev/ords/intranet/painting/ocr_dolly"
TIMEOUT = 5
SEND_COOLDOWN = 5

VIN_NUMBER = "VIN_DUMMY_001"

# =========================================================
# GLOBAL
# =========================================================
ocr_running = False
last_ocr_result = None  # ← penting dari code lama

# =========================================================
# UTIL
# =========================================================
def normalize_3digit(text):
    mapping = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1',
        'Z': '2', 'S': '5',
        'B': '8', 't': '4'
    }

    for k, v in mapping.items():
        text = text.replace(k, v)

    digits = re.sub(r'[^0-9]', '', text)

    if re.fullmatch(r'\d{3}', digits):
        return digits

    return ""


def majority_vote(history):
    if not history:
        return ""
    return Counter(history).most_common(1)[0][0]


def frame_to_base64(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(buf).decode()


# =========================================================
# API
# =========================================================
def send_to_ords(dolly_number, frame):
    img_b64 = frame_to_base64(frame)

    payload = {
        "dolly_number": dolly_number,
        "image_base64": img_b64,
        "mime_type": "image/jpeg"
    }

    try:
        r = requests.post(ORDS_URL, json=payload, timeout=TIMEOUT)
        print("[ORDS]", r.status_code, r.text)
        return r.ok
    except Exception as e:
        print("[ORDS ERROR]", e)
        return False


# =========================================================
# OCR THREAD
# =========================================================
def run_ocr(ocr, frame, history, result_dict):
    global ocr_running, last_ocr_result

    try:
        ocr_result = ocr.ocr(frame, cls=True)
        last_ocr_result = ocr_result

        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                raw_text = line[1][0]
                conf = float(line[1][1])

                text = normalize_3digit(raw_text)

                if text and conf > MIN_CONF:
                    history.append(text)
                    del history[:-HIST_MAX]

                    result_dict["text"] = majority_vote(history)
                    result_dict["conf"] = conf

    finally:
        ocr_running = False


# =========================================================
# MAIN
# =========================================================
def main():
    global ocr_running, last_ocr_result

    print("Loading OCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, use_gpu=USE_GPU)

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("Camera error")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_count = 0
    fps_time = time.time()

    history = []
    result_dict = {"text": "", "conf": 0.0}

    last_sent = ""
    last_sent_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        frame_count += 1
        annotated = frame.copy()

        do_ocr = (frame_count % OCR_INTERVAL == 0)

        # ── OCR THREAD ─────────────────────────
        if do_ocr and not ocr_running:
            ocr_running = True
            threading.Thread(
                target=run_ocr,
                args=(ocr, frame.copy(), history, result_dict),
                daemon=True
            ).start()

        # ── USE LAST OCR RESULT (FITUR LAMA) ───
        ocr_result = last_ocr_result

        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                points = line[0]
                raw_text = line[1][0]
                score = float(line[1][1])

                text = normalize_3digit(raw_text)

                if not text or score < MIN_CONF:
                    continue

                pts = np.array(
                    [(int(p[0]), int(p[1])) for p in points],
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

                x_min = int(np.min(pts[:, 0, 0]))
                y_min = int(np.min(pts[:, 0, 1]))

                cv2.putText(
                    annotated,
                    f"{text} ({score:.2f})",
                    (x_min, max(20, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

        text = result_dict["text"]

        # ── SEND API ──────────────────────────
        now = time.time()
        if text and ENABLE_API:
            if text != last_sent and (now - last_sent_time) > SEND_COOLDOWN:
                ok = send_to_ords(text, frame)

                if ok:
                    last_sent = text
                    last_sent_time = now
                    print(f"[SEND] Dolly={text}")

        # ── FPS ──────────────────────────────
        now = time.time()
        fps_disp = 1 / (now - fps_time) if (now - fps_time) > 0 else 0
        fps_time = now

        cv2.putText(
            annotated,
            f"FPS: {fps_disp:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Dolly OCR (FULL)", annotated)

        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

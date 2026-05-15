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
CAM_INDEX   = 0
OUTPUT_PATH = "output.mp4"

OCR_INTERVAL = 5
HIST_MAX     = 10

OCR_LANG        = "en"
USE_GPU         = False
MIN_CONF        = 0.5
FLIP_HORIZONTAL = False

ENABLE_API   = False
ORDS_URL     = "http://idtbintranetdev/ords/intranet/painting/ocr_dolly"
TIMEOUT      = 5
SEND_COOLDOWN = 5

# =========================================================
# STATE MACHINE SETTINGS
# =========================================================
# Berapa frame OCR harus "kosong" untuk konfirmasi dolly sudah keluar
NO_DETECT_REQUIRED = 5
# Berapa frame OCR harus baca 3 digit valid berturut-turut
# sebelum trigger session baru (READY → SCANNING)
NEW_DETECT_REQUIRED = 3
# Berapa frame raw OCR harus konsisten sebelum trigger SEND
# (menghindari false trigger dari 1 frame noise)
CONSECUTIVE_REQUIRED = 2

# =========================================================
# GLOBAL
# =========================================================
ocr_running     = False
last_ocr_result = None

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
        "mime_type":    "image/jpeg"
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

        # cari 3 digit valid dari semua line hasil OCR
        best_text = ""
        best_conf = 0.0

        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                raw_text = line[1][0]
                conf     = float(line[1][1])

                text = normalize_3digit(raw_text)

                if text and conf > MIN_CONF:
                    if conf > best_conf:
                        best_text = text
                        best_conf = conf

        if best_text:
            history.append(best_text)
            del history[:-HIST_MAX]

        # simpan raw (sebelum voting) untuk consecutive check
        result_dict["raw"]  = best_text
        result_dict["text"] = majority_vote(history)
        result_dict["conf"] = best_conf

    finally:
        ocr_running = False


# =========================================================
# HELPER: draw overlay teks tengah layar
# =========================================================
def draw_center_text(img, text, color):
    H, W = img.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness  = 3
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (W - tw) // 2
    y = (H + th) // 2
    cv2.putText(img, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y),          font, font_scale, color,     thickness)


# =========================================================
# HELPER: reset session
# =========================================================
def reset_session(history, result_dict):
    history.clear()
    result_dict["raw"]  = ""
    result_dict["text"] = ""
    result_dict["conf"] = 0.0


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
    fps_time    = time.time()

    history     = []
    result_dict = {"raw": "", "text": "", "conf": 0.0}

    last_sent_time = 0
    sent_dolly     = ""   # simpan untuk display saat WAITING_REMOVE

    # ── STATE MACHINE ──────────────────────────────────────
    # "SCANNING" | "WAITING_REMOVE" | "READY"
    scan_state         = "SCANNING"
    no_detect_counter  = 0   # counter frame OCR kosong (untuk WAITING_REMOVE)
    new_detect_counter = 0   # counter frame OCR ada (untuk READY)

    # ── consecutive raw untuk fast-confirm di SCANNING ────
    last_raw            = ""
    consecutive_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if FLIP_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        frame_count += 1
        H, W      = frame.shape[:2]
        annotated = frame.copy()
        do_ocr    = (frame_count % OCR_INTERVAL == 0)

        # jalankan OCR thread (semua state tetap OCR
        # supaya bisa deteksi "ada/tidak ada" dolly)
        if do_ocr and not ocr_running:
            ocr_running = True
            threading.Thread(
                target=run_ocr,
                args=(ocr, frame.copy(), history, result_dict),
                daemon=True
            ).start()

        # ambil nilai saat ini dari result_dict
        current_raw  = result_dict["raw"]
        current_text = result_dict["text"]
        current_conf = result_dict["conf"]
        dolly_detected = bool(current_raw)  # True kalau OCR baca 3 digit valid

        # ==================================================
        # STATE: SCANNING
        # ==================================================
        if scan_state == "SCANNING":

            # ── draw bounding box OCR result ──────────────
            if last_ocr_result and last_ocr_result[0]:
                for line in last_ocr_result[0]:
                    points   = line[0]
                    raw_text = line[1][0]
                    score    = float(line[1][1])
                    text     = normalize_3digit(raw_text)

                    if not text or score < MIN_CONF:
                        continue

                    pts = np.array(
                        [(int(p[0]), int(p[1])) for p in points],
                        dtype=np.int32
                    ).reshape((-1, 1, 2))

                    cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                    x_min = int(np.min(pts[:, 0, 0]))
                    y_min = int(np.min(pts[:, 0, 1]))
                    cv2.putText(annotated,
                                f"{text} ({score:.2f})",
                                (x_min, max(20, y_min - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # ── majority vote result ──────────────────────
            if current_text:
                cv2.putText(annotated,
                            f"Dolly: {current_text} ({current_conf:.2f})",
                            (20, H - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # ── CONSECUTIVE CHECK ─────────────────────────
            # pastikan raw OCR konsisten CONSECUTIVE_REQUIRED frame
            # sebelum di-commit ke send, menghindari false trigger noise
            if current_raw:
                if current_raw == last_raw:
                    consecutive_counter += 1
                else:
                    consecutive_counter = 0
                last_raw = current_raw
            else:
                consecutive_counter = 0
                last_raw = ""

            # ── SEND ─────────────────────────────────────
            now = time.time()
            confirmed = (consecutive_counter >= CONSECUTIVE_REQUIRED)

            if current_text and confirmed and ENABLE_API:
                if (now - last_sent_time) > SEND_COOLDOWN:
                    ok = send_to_ords(current_text, frame)
                    if ok:
                        last_sent_time = now
                        sent_dolly     = current_text
                        print(f"[SEND] Dolly={current_text}")

                        scan_state        = "WAITING_REMOVE"
                        no_detect_counter = 0
                        consecutive_counter = 0
                        last_raw = ""
                        print("[STATE] → WAITING_REMOVE")

            elif current_text and confirmed and not ENABLE_API:
                # API off — langsung pindah state untuk testing
                sent_dolly = current_text
                print(f"[API DISABLED] Dolly={current_text}")

                scan_state          = "WAITING_REMOVE"
                no_detect_counter   = 0
                consecutive_counter = 0
                last_raw            = ""
                print("[STATE] → WAITING_REMOVE")

        # ==================================================
        # STATE: WAITING_REMOVE
        # ==================================================
        elif scan_state == "WAITING_REMOVE":

            if not dolly_detected:
                no_detect_counter += 1
                if no_detect_counter >= NO_DETECT_REQUIRED:
                    scan_state        = "READY"
                    no_detect_counter = 0
                    print("[STATE] → READY")
            else:
                no_detect_counter = 0

            draw_center_text(annotated, "SENT  -  REMOVE DOLLY", (0, 220, 255))
            cv2.putText(annotated,
                        f"DOLLY: {sent_dolly}",
                        (20, H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)

        # ==================================================
        # STATE: READY
        # ==================================================
        elif scan_state == "READY":

            if dolly_detected:
                new_detect_counter += 1
                if new_detect_counter >= NEW_DETECT_REQUIRED:
                    reset_session(history, result_dict)
                    last_raw            = ""
                    consecutive_counter = 0
                    new_detect_counter  = 0
                    sent_dolly          = ""

                    scan_state = "SCANNING"
                    print("[STATE] → SCANNING (new dolly detected)")
            else:
                new_detect_counter = 0

            draw_center_text(annotated, "READY  -  INSERT DOLLY", (0, 255, 100))

        # ==================================================
        # DISPLAY UMUM
        # ==================================================
        now = time.time()
        fps_disp = 1 / (now - fps_time) if (now - fps_time) > 0 else 0
        fps_time = now

        cv2.putText(annotated, f"FPS: {fps_disp:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        state_colors = {
            "SCANNING":       (0, 255, 0),
            "WAITING_REMOVE": (0, 220, 255),
            "READY":          (0, 255, 100),
        }
        cv2.putText(annotated, f"STATE: {scan_state}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colors[scan_state], 2)

        api_status = "API: ON" if ENABLE_API else "API: OFF"
        api_color  = (0, 255, 0) if ENABLE_API else (0, 0, 255)
        cv2.putText(annotated, api_status,
                    (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, api_color, 2)

        cv2.imshow("Dolly OCR", annotated)

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

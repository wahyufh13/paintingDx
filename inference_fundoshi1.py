import cv2
import time
import requests
import base64
import json
import numpy as np
import re
import threading
from collections import Counter
from ultralytics import YOLO
from paddleocr import PaddleOCR

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "/home/suzukidx1/Downloads/model_fundoshi/best.pt"
CONF_THRES  = 0.85
IMG_SIZE    = 640
CAM_INDEX   = 0

OCR_INTERVAL = 5    # OCR tiap N frame
HIST_MAX     = 10   # panjang history untuk smoothing voting

# =========================================================
# API TOGGLE
# =========================================================
ENABLE_API        = True
ORDS_URL          = "http://172.16.173.211/ords/intranet/painting/ocr_fundoshi"
TIMEOUT_SEC       = 10
SEND_COOLDOWN_SEC = 2.0

# =========================================================
# SESSION / STABLE SETTINGS
# =========================================================
WAIT_SECONDS    = 2.0   # tunggu N detik setelah semua field ready
STABLE_REQUIRED = 10    # harus stabil N frame setelah wait time

# Jumlah frame raw OCR berturut-turut harus konsisten
# sebelum session reset di-trigger
CONSECUTIVE_REQUIRED = 2

# =========================================================
# ROI (RELATIVE TERHADAP BBOX KERTAS)
# Format: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
# =========================================================
ROI_FRAME_NO        = (0.01, 0.02, 0.79, 0.20)
ROI_CARRIERNO       = (0.80, 0.02, 0.97, 0.20)
ROI_MODEL_SPEC_DEST = (0.07, 0.23, 0.32, 0.35)
ROI_COLOR_CODE      = (0.10, 0.50, 0.59, 0.83)

VIN_ALLOWED = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$')

# =========================================================
# GLOBAL
# =========================================================
ocr_running = False


# =========================================================
# UTILITIES
# =========================================================
def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return x1, y1, x2, y2


def roi_from_bbox(px1, py1, px2, py2, roi_ratio):
    rx1, ry1, rx2, ry2 = roi_ratio
    pw = px2 - px1
    ph = py2 - py1
    x1 = px1 + int(pw * rx1)
    y1 = py1 + int(ph * ry1)
    x2 = px1 + int(pw * rx2)
    y2 = py1 + int(ph * ry2)
    return x1, y1, x2, y2


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr  = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )
    return thr


def run_paddleocr(ocr, img):
    """Return text gabungan + confidence rata-rata."""
    result = ocr.ocr(img, cls=True)
    if not result or not result[0]:
        return "", 0.0
    texts, confs = [], []
    for line in result[0]:
        texts.append(line[1][0])
        confs.append(float(line[1][1]))
    return " ".join(texts).strip(), (sum(confs) / max(len(confs), 1))


def majority_vote(history_list, min_len=1):
    """Ambil nilai paling sering muncul dari history."""
    cleaned = [x for x in history_list if x and len(x) >= min_len]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


# =========================================================
# NORMALIZATION
# =========================================================
def normalize_text(s):
    if not s:
        return ""
    s = s.strip().replace(" ", "")
    s = "".join([c for c in s if c.isalnum()])
    return s.upper()


def normalize_frame_no(raw, expect_len=17, strict_vin=False):
    s = normalize_text(raw)
    if len(s) > expect_len:
        s = s[:expect_len]
    if len(s) == expect_len:
        is_valid = bool(VIN_ALLOWED.match(s))
        if strict_vin and not is_valid:
            return ""
    return s


def normalize_carrier(raw):
    s = normalize_text(raw)
    digits = "".join([c for c in s if c.isdigit()])
    if not digits:
        return ""
    if len(digits) > 3:
        digits = digits[:3]
    return digits.zfill(3)


# =========================================================
# API
# =========================================================
def frame_to_base64_jpg(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def send_to_ords(vin_number, dolly_number, paper_crop, color_code, model_spec):
    """Kirim hasil OCR + gambar ke ORDS. Hanya dipanggil jika ENABLE_API = True."""
    img_b64 = frame_to_base64_jpg(paper_crop)
    payload = {
        "vin_number":       vin_number,
        "dolly_number":     dolly_number,
        "topcoat_cc":       color_code,
        "model_spec_dest":  model_spec,
        "image_base64":     img_b64,
        "mime_type":        "image/jpeg"
    }
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(ORDS_URL, headers=headers,
                          data=json.dumps(payload), timeout=TIMEOUT_SEC)
        print("ORDS status:", r.status_code)
        print("ORDS response:", r.text)
        return r.ok
    except Exception as e:
        print("[ORDS] ERROR:", str(e))
        return False


# =========================================================
# OCR THREAD
# =========================================================
def run_ocr(ocr,
            crop_frame, crop_carrier, crop_color, crop_model,
            frame_hist, carrier_hist, colorcode_hist, modelspec_hist,
            result_dict):
    """
    Jalankan OCR untuk semua 4 ROI dalam satu thread.
    Hasil raw dan voting ditulis ke result_dict.
    """
    global ocr_running
    try:
        # --- Frame No ---
        txt_frame, conf_frame = run_paddleocr(ocr, preprocess_for_ocr(crop_frame))
        txt_frame = normalize_frame_no(txt_frame, expect_len=17, strict_vin=False)
        if txt_frame:
            frame_hist.append(txt_frame)
            del frame_hist[:-HIST_MAX]
        # simpan raw (sebelum voting) untuk deteksi session change
        result_dict["frame_raw"]  = txt_frame
        result_dict["frame"]      = majority_vote(frame_hist, min_len=10)
        result_dict["conf_frame"] = conf_frame

        # --- Carrier No ---
        txt_carrier, conf_carrier = run_paddleocr(ocr, preprocess_for_ocr(crop_carrier))
        txt_carrier = normalize_carrier(txt_carrier)
        if txt_carrier:
            carrier_hist.append(txt_carrier)
            del carrier_hist[:-HIST_MAX]
        result_dict["carrier"]      = majority_vote(carrier_hist, min_len=3)
        result_dict["conf_carrier"] = conf_carrier

        # --- Color Code ---
        txt_color, conf_color = run_paddleocr(ocr, preprocess_for_ocr(crop_color))
        txt_color = normalize_text(txt_color)
        if txt_color:
            colorcode_hist.append(txt_color)
            del colorcode_hist[:-HIST_MAX]
        result_dict["color"]      = majority_vote(colorcode_hist, min_len=2)
        result_dict["conf_color"] = conf_color

        # --- Model Spec Dest ---
        txt_model, conf_model = run_paddleocr(ocr, preprocess_for_ocr(crop_model))
        txt_model = normalize_text(txt_model)
        if txt_model:
            modelspec_hist.append(txt_model)
            del modelspec_hist[:-HIST_MAX]
        result_dict["model"]      = majority_vote(modelspec_hist, min_len=3)
        result_dict["conf_model"] = conf_model

    finally:
        ocr_running = False


# =========================================================
# MAIN
# =========================================================
def main():
    global ocr_running

    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)

    print("Loading PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Camera not detected")
        return

    print(f"[API] ENABLE_API = {ENABLE_API}")

    frame_count = 0
    fps_time    = time.time()

    # ── history smoothing ──────────────────────────────────
    frame_hist     = []
    carrier_hist   = []
    colorcode_hist = []
    modelspec_hist = []

    # ── cache hasil OCR (display + logic) ─────────────────
    result_dict = {
        "frame_raw":    "",
        "frame":        "",
        "carrier":      "",
        "color":        "",
        "model":        "",
        "conf_frame":   0.0,
        "conf_carrier": 0.0,
        "conf_color":   0.0,
        "conf_model":   0.0,
    }

    # ── session management ─────────────────────────────────
    current_vin_session  = ""
    session_start_time   = 0
    stable_frame_counter = 0
    session_sent         = False

    # ── consecutive raw VIN counter (untuk fast reset) ────
    last_raw_vin        = ""
    consecutive_new_vin = 0

    # ── anti-duplicate ─────────────────────────────────────
    last_sent_vin   = ""
    last_sent_dolly = ""
    last_sent_time  = 0

    # ── last confirmed values (untuk display & send) ───────
    last_frame_no   = ""
    last_carrier_no = ""
    last_color_code = ""
    last_model_spec = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame error")
            break

        frame_count += 1
        H, W = frame.shape[:2]

        results   = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
        annotated = frame.copy()
        r         = results[0]
        do_ocr    = (frame_count % OCR_INTERVAL == 0)

        if r.boxes is not None and len(r.boxes) > 0:

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            best  = int(np.argmax(confs))

            px1, py1, px2, py2 = boxes[best]
            px1, py1, px2, py2 = clamp_box(px1, py1, px2, py2, W, H)

            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 3)

            # ── Hitung semua ROI ───────────────────────────
            fx1,  fy1,  fx2,  fy2  = clamp_box(*roi_from_bbox(px1, py1, px2, py2, ROI_FRAME_NO),        W, H)
            cx1,  cy1,  cx2,  cy2  = clamp_box(*roi_from_bbox(px1, py1, px2, py2, ROI_CARRIERNO),       W, H)
            ccx1, ccy1, ccx2, ccy2 = clamp_box(*roi_from_bbox(px1, py1, px2, py2, ROI_COLOR_CODE),      W, H)
            msx1, msy1, msx2, msy2 = clamp_box(*roi_from_bbox(px1, py1, px2, py2, ROI_MODEL_SPEC_DEST), W, H)

            # ── Gambar semua ROI ───────────────────────────
            roi_clr = (255, 200, 0)
            cv2.rectangle(annotated, (fx1,  fy1),  (fx2,  fy2),  roi_clr, 2)
            cv2.rectangle(annotated, (cx1,  cy1),  (cx2,  cy2),  roi_clr, 2)
            cv2.rectangle(annotated, (ccx1, ccy1), (ccx2, ccy2), roi_clr, 2)
            cv2.rectangle(annotated, (msx1, msy1), (msx2, msy2), roi_clr, 2)

            # ── OCR THREAD ────────────────────────────────
            if do_ocr and not ocr_running:
                crop_frame   = frame[fy1:fy2,   fx1:fx2  ].copy()
                crop_carrier = frame[cy1:cy2,   cx1:cx2  ].copy()
                crop_color   = frame[ccy1:ccy2, ccx1:ccx2].copy()
                crop_model   = frame[msy1:msy2, msx1:msx2].copy()

                ocr_running = True
                threading.Thread(
                    target=run_ocr,
                    args=(ocr,
                          crop_frame, crop_carrier, crop_color, crop_model,
                          frame_hist, carrier_hist, colorcode_hist, modelspec_hist,
                          result_dict),
                    daemon=True
                ).start()

            # ── FAST SESSION RESET (dari raw OCR) ─────────
            # Tidak nunggu majority vote — deteksi perubahan VIN
            # dari raw result, tapi harus konsisten CONSECUTIVE_REQUIRED
            # frame berturut-turut untuk menghindari false trigger dari
            # 1 frame noise/blur
            candidate_vin_raw = result_dict.get("frame_raw", "")

            if candidate_vin_raw and len(candidate_vin_raw) >= 15:
                if candidate_vin_raw == last_raw_vin:
                    # raw konsisten dengan frame sebelumnya
                    if candidate_vin_raw != current_vin_session:
                        # dan berbeda dari session aktif → tambah counter
                        consecutive_new_vin += 1
                else:
                    # raw berubah → reset counter
                    consecutive_new_vin = 0

                last_raw_vin = candidate_vin_raw

                if consecutive_new_vin >= CONSECUTIVE_REQUIRED:
                    # ✅ VIN baru sudah stabil N frame → trigger reset
                    print(f"[SESSION] VIN changed: {current_vin_session} → {candidate_vin_raw} "
                          f"(confirmed {consecutive_new_vin} frames)")

                    current_vin_session = candidate_vin_raw

                    # clear SEMUA history termasuk frame_hist
                    frame_hist.clear()
                    carrier_hist.clear()
                    colorcode_hist.clear()
                    modelspec_hist.clear()

                    # reset last confirmed values
                    last_frame_no   = candidate_vin_raw   # langsung pakai VIN baru
                    last_carrier_no = ""
                    last_color_code = ""
                    last_model_spec = ""

                    # reset result_dict field lain
                    result_dict["frame"]   = candidate_vin_raw
                    result_dict["carrier"] = ""
                    result_dict["color"]   = ""
                    result_dict["model"]   = ""

                    # reset session state
                    stable_frame_counter = 0
                    session_start_time   = 0
                    session_sent         = False

                    # reset consecutive counter setelah reset berhasil
                    consecutive_new_vin = 0
            else:
                # raw kosong atau terlalu pendek → reset consecutive
                consecutive_new_vin = 0
                last_raw_vin        = ""

            # ── update last confirmed values dari result_dict ─
            if result_dict["frame"]:
                last_frame_no = result_dict["frame"]
            if result_dict["carrier"]:
                last_carrier_no = result_dict["carrier"]
            if result_dict["color"]:
                last_color_code = result_dict["color"]
            if result_dict["model"]:
                last_model_spec = result_dict["model"]

            # ── STABLE COUNTER + SEND TO ORDS ─────────────
            vin_ok    = (len(last_frame_no)   == 17)
            dolly_ok  = (len(last_carrier_no) == 3)
            all_ready = vin_ok and dolly_ok

            if all_ready:
                now_ts = time.time()

                # mulai timer jika belum
                if session_start_time == 0:
                    session_start_time = now_ts
                    print("[SESSION] All fields ready, start wait timer")

                # hitung stable frame setelah wait time terpenuhi
                if (now_ts - session_start_time) >= WAIT_SECONDS:
                    stable_frame_counter += 1
                    print(f"[STABLE] {stable_frame_counter}/{STABLE_REQUIRED}")

                is_cooldown_ok = (now_ts - last_sent_time) >= SEND_COOLDOWN_SEC

                if (is_cooldown_ok
                        and stable_frame_counter >= STABLE_REQUIRED
                        and not session_sent):

                    if ENABLE_API:
                        paper_crop = frame[py1:py2, px1:px2]
                        ok = send_to_ords(last_frame_no, last_carrier_no,
                                          paper_crop, last_color_code, last_model_spec)
                        if ok:
                            session_sent    = True
                            last_sent_vin   = last_frame_no
                            last_sent_dolly = last_carrier_no
                            last_sent_time  = now_ts
                            print(f"[ORDS] SENT: VIN={last_frame_no}, DOLLY={last_carrier_no}")
                        else:
                            print("[ORDS] FAILED (response not OK)")
                    else:
                        session_sent = True
                        print(f"[API DISABLED] VIN={last_frame_no} | DOLLY={last_carrier_no} "
                              f"| COLOR={last_color_code} | MODEL={last_model_spec}")

            else:
                # kalau field tidak lengkap, reset timer & counter
                session_start_time   = 0
                stable_frame_counter = 0

            # ── DISPLAY OCR RESULT (dekat bbox kertas) ────
            cv2.putText(annotated,
                        f"Frame No:   {last_frame_no or '-'} ({result_dict['conf_frame']:.2f})",
                        (px1, max(30, py1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(annotated,
                        f"Carrier No: {last_carrier_no or '-'} ({result_dict['conf_carrier']:.2f})",
                        (px1, max(30, py1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(annotated,
                        f"Color Code: {last_color_code or '-'} ({result_dict['conf_color']:.2f})",
                        (px1, py2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            cv2.putText(annotated,
                        f"Model Spec: {last_model_spec or '-'} ({result_dict['conf_model']:.2f})",
                        (px1, py2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            # ── Status API & session (pojok kiri bawah) ───
            api_status = "API: ON" if ENABLE_API else "API: OFF"
            api_color  = (0, 255, 0) if ENABLE_API else (0, 0, 255)
            cv2.putText(annotated, api_status,
                        (20, H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, api_color, 2)

            sent_status = "SENT v" if session_sent else f"STABLE: {stable_frame_counter}/{STABLE_REQUIRED}"
            cv2.putText(annotated, sent_status,
                        (20, H - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        # ── FPS ───────────────────────────────────────────
        now = time.time()
        fps = 1 / (now - fps_time) if (now - fps_time) > 0 else 0
        fps_time = now

        cv2.putText(annotated, f"FPS: {fps:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Jetson YOLO + OCR", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

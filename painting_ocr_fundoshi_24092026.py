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
# API
# =========================================================
ENABLE_API        = True
ORDS_URL          = "http://172.16.173.211/ords/intranet/painting/ocr_fundoshi"
TIMEOUT_SEC       = 10
SEND_COOLDOWN_SEC = 2.0

# =========================================================
# SESSION / STABLE SETTINGS
# =========================================================
WAIT_SECONDS    = 2.0   # tunggu N detik setelah semua field ready
STABLE_REQUIRED = 60    # harus stabil N frame setelah wait time

# =========================================================
# STATE MACHINE SETTINGS
# =========================================================
# Berapa frame YOLO harus "kosong" untuk konfirmasi kertas diangkat
NO_DETECT_REQUIRED  = 20
# Berapa frame YOLO harus "ada" untuk konfirmasi kertas baru masuk
NEW_DETECT_REQUIRED = 10

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
    result = ocr.ocr(img, cls=True)
    if not result or not result[0]:
        return "", 0.0
    texts, confs = [], []
    for line in result[0]:
        texts.append(line[1][0])
        confs.append(float(line[1][1]))
    return " ".join(texts).strip(), (sum(confs) / max(len(confs), 1))


def majority_vote(history_list, min_len=1):
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
    global ocr_running
    try:
        # --- Frame No ---
        txt_frame, conf_frame = run_paddleocr(ocr, preprocess_for_ocr(crop_frame))
        txt_frame = normalize_frame_no(txt_frame, expect_len=17, strict_vin=False)
        if txt_frame:
            frame_hist.append(txt_frame)
            del frame_hist[:-HIST_MAX]
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
# HELPER: reset semua session state
# =========================================================
def reset_session(frame_hist, carrier_hist, colorcode_hist, modelspec_hist, result_dict):
    frame_hist.clear()
    carrier_hist.clear()
    colorcode_hist.clear()
    modelspec_hist.clear()

    result_dict["frame"]        = ""
    result_dict["carrier"]      = ""
    result_dict["color"]        = ""
    result_dict["model"]        = ""
    result_dict["conf_frame"]   = 0.0
    result_dict["conf_carrier"] = 0.0
    result_dict["conf_color"]   = 0.0
    result_dict["conf_model"]   = 0.0

    return {
        "last_frame_no":        "",
        "last_carrier_no":      "",
        "last_color_code":      "",
        "last_model_spec":      "",
        "session_start_time":   0,
        "stable_frame_counter": 0,
        "session_sent":         False,
    }


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
    # shadow biar terbaca di background apapun
    cv2.putText(img, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y),          font, font_scale, color,     thickness)


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

    # ── cache hasil OCR ────────────────────────────────────
    result_dict = {
        "frame":        "",
        "carrier":      "",
        "color":        "",
        "model":        "",
        "conf_frame":   0.0,
        "conf_carrier": 0.0,
        "conf_color":   0.0,
        "conf_model":   0.0,
    }

    # ── session ────────────────────────────────────────────
    last_frame_no        = ""
    last_carrier_no      = ""
    last_color_code      = ""
    last_model_spec      = ""
    session_start_time   = 0
    stable_frame_counter = 0
    session_sent         = False
    last_sent_time       = 0

    # ── STATE MACHINE ──────────────────────────────────────
    # "SCANNING" → "WAITING_REMOVE" → "READY" → "SCANNING" → ...
    scan_state         = "SCANNING"
    no_detect_counter  = 0
    new_detect_counter = 0

    # ── simpan hasil terakhir untuk display saat WAITING_REMOVE ─
    sent_frame_no   = ""
    sent_carrier_no = ""
    sent_color_code = ""
    sent_model_spec = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame error")
            break

        frame_count += 1
        H, W = frame.shape[:2]

        results        = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
        annotated      = frame.copy()
        r              = results[0]
        do_ocr         = (frame_count % OCR_INTERVAL == 0)
        paper_detected = (r.boxes is not None and len(r.boxes) > 0)

        # ==================================================
        # STATE: SCANNING
        # ==================================================
        if scan_state == "SCANNING":

            if paper_detected:
                no_detect_counter = 0

                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                best  = int(np.argmax(confs))

                px1, py1, px2, py2 = boxes[best]
                px1, py1, px2, py2 = clamp_box(px1, py1, px2, py2, W, H)

                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 3)

                # ── ROI ───────────────────────────────────
                fx1,  fy1,  fx2,  fy2  = clamp_box(*roi_from_bbox(px1, py1, px2, py2, ROI_FRAME_NO),        W, H)
                cx1,  cy1,  cx2,  cy2  = clamp_box(*roi_from_bbox(px1, py1, px2, py2, ROI_CARRIERNO),       W, H)
                ccx1, ccy1, ccx2, ccy2 = clamp_box(*roi_from_bbox(px1, py1, px2, py2, ROI_COLOR_CODE),      W, H)
                msx1, msy1, msx2, msy2 = clamp_box(*roi_from_bbox(px1, py1, px2, py2, ROI_MODEL_SPEC_DEST), W, H)

                roi_clr = (255, 200, 0)
                cv2.rectangle(annotated, (fx1,  fy1),  (fx2,  fy2),  roi_clr, 2)
                cv2.rectangle(annotated, (cx1,  cy1),  (cx2,  cy2),  roi_clr, 2)
                cv2.rectangle(annotated, (ccx1, ccy1), (ccx2, ccy2), roi_clr, 2)
                cv2.rectangle(annotated, (msx1, msy1), (msx2, msy2), roi_clr, 2)

                # ── OCR THREAD ────────────────────────────
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

                # ── update last confirmed values ──────────
                if result_dict["frame"]:
                    last_frame_no = result_dict["frame"]
                if result_dict["carrier"]:
                    last_carrier_no = result_dict["carrier"]
                if result_dict["color"]:
                    last_color_code = result_dict["color"]
                if result_dict["model"]:
                    last_model_spec = result_dict["model"]

                # ── STABLE COUNTER + SEND ─────────────────
                vin_ok    = (len(last_frame_no)   == 17)
                dolly_ok  = (len(last_carrier_no) == 3)
                all_ready = vin_ok and dolly_ok

                if all_ready:
                    now_ts = time.time()

                    if session_start_time == 0:
                        session_start_time = now_ts
                        print("[SESSION] All fields ready, start wait timer")

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
                                session_sent   = True
                                last_sent_time = now_ts
                                print(f"[ORDS] SENT: VIN={last_frame_no}, DOLLY={last_carrier_no}")

                                sent_frame_no   = last_frame_no
                                sent_carrier_no = last_carrier_no
                                sent_color_code = last_color_code
                                sent_model_spec = last_model_spec

                                scan_state        = "WAITING_REMOVE"
                                no_detect_counter = 0
                                print("[STATE] → WAITING_REMOVE")
                            else:
                                print("[ORDS] FAILED (response not OK)")
                        else:
                            session_sent = True
                            print(f"[API DISABLED] VIN={last_frame_no} | DOLLY={last_carrier_no} "
                                  f"| COLOR={last_color_code} | MODEL={last_model_spec}")

                            sent_frame_no   = last_frame_no
                            sent_carrier_no = last_carrier_no
                            sent_color_code = last_color_code
                            sent_model_spec = last_model_spec

                            scan_state        = "WAITING_REMOVE"
                            no_detect_counter = 0
                            print("[STATE] → WAITING_REMOVE")

                else:
                    session_start_time   = 0
                    stable_frame_counter = 0

                # ── display OCR result ────────────────────
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

                # stable progress bar
                bar_w = min(int((stable_frame_counter / STABLE_REQUIRED) * (px2 - px1)), px2 - px1)
                cv2.rectangle(annotated, (px1, py2 + 60), (px1 + bar_w, py2 + 70), (0, 200, 255), -1)
                cv2.rectangle(annotated, (px1, py2 + 60), (px2,         py2 + 70), (100, 100, 100), 1)

            else:
                no_detect_counter += 1

        # ==================================================
        # STATE: WAITING_REMOVE
        # ==================================================
        elif scan_state == "WAITING_REMOVE":

            if not paper_detected:
                no_detect_counter += 1
                if no_detect_counter >= NO_DETECT_REQUIRED:
                    scan_state        = "READY"
                    no_detect_counter = 0
                    print("[STATE] → READY")
            else:
                no_detect_counter = 0

            draw_center_text(annotated, "SENT  -  REMOVE PAPER", (0, 220, 255))

            cv2.putText(annotated,
                        f"VIN   : {sent_frame_no}",
                        (20, H - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.putText(annotated,
                        f"DOLLY : {sent_carrier_no}   COLOR : {sent_color_code}   MODEL : {sent_model_spec}",
                        (20, H - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        # ==================================================
        # STATE: READY
        # ==================================================
        elif scan_state == "READY":

            if paper_detected:
                new_detect_counter += 1
                if new_detect_counter >= NEW_DETECT_REQUIRED:
                    sess = reset_session(
                        frame_hist, carrier_hist, colorcode_hist, modelspec_hist,
                        result_dict
                    )
                    last_frame_no        = sess["last_frame_no"]
                    last_carrier_no      = sess["last_carrier_no"]
                    last_color_code      = sess["last_color_code"]
                    last_model_spec      = sess["last_model_spec"]
                    session_start_time   = sess["session_start_time"]
                    stable_frame_counter = sess["stable_frame_counter"]
                    session_sent         = sess["session_sent"]

                    new_detect_counter = 0
                    scan_state         = "SCANNING"
                    print("[STATE] → SCANNING (new paper detected)")
            else:
                new_detect_counter = 0

            draw_center_text(annotated, "READY  -  INSERT PAPER", (0, 255, 100))

        # ==================================================
        # DISPLAY UMUM (semua state)
        # ==================================================
        now = time.time()
        fps = 1 / (now - fps_time) if (now - fps_time) > 0 else 0
        fps_time = now
        cv2.putText(annotated, f"FPS: {fps:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        state_colors = {
            "SCANNING":       (0, 255, 0),
            "WAITING_REMOVE": (0, 220, 255),
            "READY":          (0, 255, 100),
        }
        cv2.putText(annotated, f"STATE: {scan_state}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colors[scan_state], 2)

        api_status = "API: ON" if ENABLE_API else "API: OFF"
        api_color  = (0, 255, 0) if ENABLE_API else (0, 0, 255)
        cv2.putText(annotated, api_status,
                    (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, api_color, 2)

        cv2.imshow("Jetson YOLO + OCR", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

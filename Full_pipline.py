# ============================================
# automate.py — record one clip -> process it
# ============================================

# ---------- Standard & system ----------
import os, re, cv2, math, json, shutil, time, subprocess
from datetime import datetime
from glob import glob
from pathlib import Path
from collections import defaultdict

# ---------- Third-party ----------
import numpy as np
import torch, librosa, pandas as pd, soundfile as sf
from PIL import Image
import torch.nn as nn
from timm import create_model
from torchvision import transforms
from pydub import AudioSegment
from ultralytics import YOLO

# Headless OpenCV shim to avoid ultralytics import issues
if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0
    cv2.destroyAllWindows = lambda *a, **k: None

# ---------- GPIO ----------
from gpiozero import LED, Button

# =========================
#        CONFIG
# =========================
# Recording (input) -> we write each clip here, then process it immediately
REC_VIDEO_PATTERN = "/home/gasemissions/record_{ts}.mp4"
AUDIO_DEV = "hw:3,0"          # ALSA device for mic (change if needed)
REC_DURATION = 10             # seconds per clip
VIDEO_SIZE = "1280x720"       # '1920x1080' etc.
VIDEO_FPS  = "10"

# Processing (output) root
OUTPUT_BASE_DIRECTORY = "/home/gasemissions/Processed_Videos"

# Models / metadata
PRED_MODEL_WEIGHTS       = "/home/gasemissions/models/model_weights.pth"
CLASS_MAP_JSON           = "/home/gasemissions/models/idx_to_names.json"
ALLOWED_JSON             = "/home/gasemissions/models/name_to_indices.json"
IDX_TO_SINGLE_LABEL_JSON = "/home/gasemissions/models/idx_to_single_label.json"

# Emissions CSV
CSV_PATH          = "/home/gasemissions/data/filtered_unique.csv"
CSV_SEARCH_COLUMN = "Cn"
CSV_RETURN_COLUMN = "Ewltp (g/km)"

# Audio helper config
SAMPLE_RATE = 22050
N_MFCC = 13
APPLY_PREEMPHASIS = True

# Button & LEDs
PIN_RED = 26
PIN_GREEN = 6
PIN_BLUE = 5
PIN_BUTTON = 27

# Sound trigger (optional)
TEMP_WAV = "/home/gasemissions/temp.wav"
THRESHOLD_DBFS = -30.0  # clap threshold; adjust as needed

# =========================
#      GPIO Helpers
# =========================
red   = LED(PIN_RED)
green = LED(PIN_GREEN)
blue  = LED(PIN_BLUE)
button = Button(PIN_BUTTON, bounce_time=0.3)

def set_led(r, g, b):
    red.value = r
    green.value = g
    blue.value = b

# =========================
#   Recording functions
# =========================
def detect_loud_sound():
    """Record ~0.8s audio, return True if loud (dBFS > threshold)."""
    subprocess.run([
        'ffmpeg', '-hide_banner', '-nostats',
        '-f', 'alsa', '-i', AUDIO_DEV,
        '-t', '0.8', '-y', TEMP_WAV
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(TEMP_WAV) or os.path.getsize(TEMP_WAV) < 1000:
        return False
    audio = AudioSegment.from_wav(TEMP_WAV)
    return audio.dBFS > THRESHOLD_DBFS

def record_one_clip():
    """Record one video+audio clip with ffmpeg; return the filename path."""
    os.makedirs(os.path.dirname(OUTPUT_BASE_DIRECTORY), exist_ok=True)  # ensure /home/gasemissions exists
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REC_VIDEO_PATTERN.format(ts=ts)
    print(f"[REC] Recording -> {out_path}")

    set_led(0, 1, 1)  # cyan while recording
    try:
        subprocess.run([
            'ffmpeg',
            # Video
            '-f', 'v4l2',
            '-video_size', VIDEO_SIZE,
            '-framerate', VIDEO_FPS,
            '-i', '/dev/video0',
            # Audio
            '-f', 'alsa',
            '-i', AUDIO_DEV,
            # Duration + overwrite
            '-t', str(REC_DURATION),
            '-y',
            out_path
        ])
    finally:
        set_led(1, 0, 1)  # magenta idle
    print(f"[REC] Done -> {out_path}")
    return out_path

# =========================
#   Pipeline: helpers
# =========================
def extract_audio(video_path, audio_output_path):
    """Extract audio with ffmpeg (mono 44.1k)."""
    try:
        subprocess.run([
            'ffmpeg','-hide_banner','-nostats',
            '-i', video_path,
            '-vn',          # no video
            '-ac','1',      # mono
            '-ar','44100',  # 44.1 kHz
            '-y', audio_output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        print(f"Error extracting audio with ffmpeg: {e}")
        return False

def extract_audio_segment(center_sec, range_sec, sample_rate, mono_audio, dest_path):
    center_sample = int(center_sec * sample_rate)
    range_samples = int(range_sec * sample_rate)
    start_sample = max(0, center_sample - range_samples)
    end_sample = min(len(mono_audio), center_sample + range_samples)
    audio_segment = mono_audio[start_sample:end_sample]
    sf.write(dest_path, audio_segment, sample_rate)

def extract_segments(audio_path, onsets, output_dir, source_name, range_sec=1):
    if not os.path.exists(audio_path):
        print(f"Skipping (audio file not found): {audio_path}")
        return
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if len(y) == 0:
            print(f"Skipping (empty audio): {audio_path}")
            return
        os.makedirs(output_dir, exist_ok=True)
        for i, center_sec in enumerate(onsets):
            output_path = os.path.join(output_dir, f"{source_name}_segment{i+1}_{center_sec:.2f}s.wav")
            extract_audio_segment(center_sec, range_sec, sr, y, output_path)
        print(f"Saved {len(onsets)} segments to {output_dir}")
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")

# =========================
#   Pipeline: core steps
# =========================
def process_video_with_car_capture(
    input_video_path,
    output_video_path,
    left_line_x,
    right_line_x,
    output_image_dir='car_captures',
    model_name='yolov8n.pt',
    detection_interval=5,
    min_confidence=0.5,
    min_car_width=100,
    min_car_height=100,
    road_distance=4
):
    model = YOLO(model_name, verbose=False)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0

    os.makedirs(output_image_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    tracked_cars = {}
    car_id_counter = 0
    frame_count = 0
    images = []
    crossing_durations = []

    yellow_left_x  = right_line_x
    yellow_right_x = frame_width

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps
        frame_copy = frame.copy()

        # Draw lines
        cv2.line(frame_copy, (left_line_x, 0), (left_line_x, frame_height), (255, 0, 0), 2)
        cv2.line(frame_copy, (right_line_x, 0), (right_line_x, frame_height), (255, 0, 0), 2)
        cv2.line(frame_copy, (yellow_left_x, 0), (yellow_left_x, frame_height), (0, 255, 255), 2)
        cv2.line(frame_copy, (yellow_right_x, 0), (yellow_right_x, frame_height), (0, 255, 255), 2)
        cv2.putText(frame_copy, f"Time: {current_time:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if frame_count % detection_interval == 0:
            results = model(frame, verbose=False)
            current_frame_cars = {}
            all_detections = []

            for result in results:
                for box in result.boxes:
                    # COCO: 2 = car (if using yolov8n.pt default)
                    if int(box.cls) == 2:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        center_x = (x1 + x2) // 2
                        if (conf >= min_confidence and
                            (x2 - x1) >= min_car_width and
                            (y2 - y1) >= min_car_height):
                            all_detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'center_x': center_x,
                                'conf': conf
                            })

            # track existing cars
            for car_id, car_data in tracked_cars.items():
                best_match, best_distance = None, float('inf')
                for i, detection in enumerate(all_detections):
                    x1, y1, x2, y2 = detection['bbox']
                    center_x = detection['center_x']
                    distance = abs(center_x - car_data['prev_x']) + abs(y1 - car_data['prev_y'])
                    if distance < 150 and distance < best_distance:
                        best_match, best_distance = i, distance

                if best_match is not None:
                    detection = all_detections.pop(best_match)
                    x1, y1, x2, y2 = detection['bbox']
                    center_x = detection['center_x']
                    conf = detection['conf']
                    direction = car_data['direction'] or ('right' if center_x > car_data['prev_x'] else 'left')

                    if conf > car_data['best_conf']:
                        best_conf, best_frame, best_bbox = conf, frame[y1:y2, x1:x2], (x1, y1, x2, y2)
                    else:
                        best_conf = car_data['best_conf']
                        best_frame = car_data['best_frame']
                        best_bbox = car_data['best_bbox']

                    cross = car_data['crossing_data']
                    if cross['enter_time'] is None and left_line_x < center_x < right_line_x:
                        cross['enter_time'] = current_time
                    if cross['exit_time'] is None and yellow_left_x < center_x < yellow_right_x:
                        cross['exit_time'] = current_time

                    current_frame_cars[car_id] = {
                        'prev_x': center_x,
                        'prev_y': y1,
                        'direction': direction,
                        'best_conf': best_conf,
                        'best_frame': best_frame,
                        'best_bbox': best_bbox,
                        'crossing_data': cross
                    }

            # new cars
            for detection in all_detections:
                x1, y1, x2, y2 = detection['bbox']
                center_x = detection['center_x']
                conf = detection['conf']

                if left_line_x < center_x < right_line_x:
                    car_id = car_id_counter
                    car_id_counter += 1
                    print(f"New car {car_id} at {current_time:.2f}s (conf: {conf:.2f})")
                    current_frame_cars[car_id] = {
                        'prev_x': center_x,
                        'prev_y': y1,
                        'direction': None,
                        'best_conf': conf,
                        'best_frame': frame[y1:y2, x1:x2],
                        'best_bbox': (x1, y1, x2, y2),
                        'crossing_data': {
                            'enter_time': current_time,
                            'exit_time': None
                        }
                    }

            # finalize cars that disappeared
            for car_id, car_data in list(tracked_cars.items()):
                if car_id not in current_frame_cars:
                    cross = car_data['crossing_data']
                    if cross['enter_time'] and cross['exit_time']:
                        duration = cross['exit_time'] - cross['enter_time']
                        if duration > 0:
                            velocity = (road_distance / duration) * 3.6
                            if car_data['direction'] == 'right':
                                filename = f"{output_image_dir}/car_{car_id}_{velocity:.2f}kmh.jpg"
                                cv2.imwrite(filename, car_data['best_frame'])
                                images.append(filename)
                            crossing_durations.append({
                                'car_id': car_id,
                                'duration': duration,
                                'direction': car_data['direction'],
                                'start_time': cross['enter_time'],
                                'end_time': cross['exit_time'],
                                'velocity': math.ceil(velocity)
                            })
                            print(f"Car done. Duration: {duration:.2f}s | Velocity: {velocity:.2f} km/h")

            tracked_cars = current_frame_cars

        # draw best boxes
        for car_id, car_data in tracked_cars.items():
            x1, y1, x2, y2 = car_data['best_bbox']
            center_x = (x1 + x2) // 2
            cross = car_data['crossing_data']
            color = (0, 255, 0) if left_line_x < center_x < right_line_x else (0, 0, 255)
            if cross['enter_time'] and cross['exit_time']:
                color = (255, 255, 255)

            label = f"C:{car_data['best_conf']:.2f} Dir:{car_data['direction'] or '?'}"
            if cross['enter_time']:
                label += f" IN:{cross['enter_time']:.2f}s"
            if cross['exit_time'] and cross['enter_time']:
                duration = cross['exit_time'] - cross['enter_time']
                if duration > 0:
                    velocity = (road_distance / duration) * 3.6
                    label += f" {velocity:.1f}km/h"

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame_copy)

    cap.release()
    out.release()

    print(f"\nSaved car images to: {output_image_dir}")
    print(f"Processed video saved to: {output_video_path}")
    return images, crossing_durations

# =========================
#   Prediction pipeline
# =========================
def load_class_map(class_map_path):
    with open(class_map_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    def is_intlike(x):
        try: int(x); return True
        except: return False
    if isinstance(obj, list):
        idx_to_class = {i: str(v) for i, v in enumerate(obj)}
        class_to_idx = {v: k for k, v in idx_to_class.items()}
        return class_to_idx, idx_to_class
    if isinstance(obj, dict):
        if all(is_intlike(k) for k in obj.keys()):
            idx_to_class = {}
            for k, v in obj.items():
                i = int(k)
                idx_to_class[i] = str(v["class"]) if isinstance(v, dict) and "class" in v else str(v)
            class_to_idx = {v: k for k, v in idx_to_class.items()}
            return class_to_idx, idx_to_class
        else:
            class_to_idx = {str(k): int(v) for k, v in obj.items()}
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            return class_to_idx, idx_to_class
    raise ValueError("Unsupported class_map JSON")

def load_allowed_indices(allowed_json_path, class_to_idx, idx_to_class):
    with open(allowed_json_path, "r", encoding="utf-8") as f:
        allowed = json.load(f)
    allowed_idx = set()
    if isinstance(allowed, dict):
        for _, lst in allowed.items():
            if isinstance(lst, list):
                for it in lst:
                    try: allowed_idx.add(int(it))
                    except: pass
    elif isinstance(allowed, list):
        tokens = [t.lower().strip() for t in allowed if isinstance(t, str) and t.strip()]
        for idx, name in idx_to_class.items():
            if any(tok in name.lower() for tok in tokens):
                allowed_idx.add(int(idx))
    else:
        raise ValueError("allowed_json must be dict or list")
    valid = set(int(k) for k in idx_to_class.keys())
    return sorted(i for i in allowed_idx if i in valid)

def load_idx_to_single_label(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {int(k): str(v) for k, v in obj.items()}

def load_model(model_weights_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("efficientnetv2_m", pretrained=False, num_classes=num_classes)
    old_conv = model.conv_stem
    model.conv_stem = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    state = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model, device

_transform_gray224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

@torch.no_grad()
def predict_image(image_path, model, device, allowed_indices=None):
    img = Image.open(image_path).convert("L")
    x = _transform_gray224(img).unsqueeze(0).to(device)
    logits = model(x).squeeze(0)
    if allowed_indices:
        mask = torch.full_like(logits, float("-inf"))
        mask[allowed_indices] = 0.0
        logits = logits + mask
    return int(torch.argmax(logits).item())

def find_emissions_by_short_label(short_label, df, search_column, return_column):
    if df is None or df.empty or short_label is None:
        return None
    s = str(short_label).lower().strip()
    if not s:
        return None
    col = df[search_column].astype(str).str.lower().str.strip()
    m = (col == s)
    if m.any():
        val = pd.to_numeric(df.loc[m, return_column], errors="coerce").iloc[0]
        return None if pd.isna(val) else float(val)
    m = col.str.contains(s, na=False)
    if m.any():
        val = pd.to_numeric(df.loc[m, return_column], errors="coerce").iloc[0]
        return None if pd.isna(val) else float(val)
    return None

def predict_all_car_captures_and_GasEmissions(
    base_dir,
    model_weights_path,
    class_map_path,
    allowed_json_path,
    csv_path=CSV_PATH,
    search_column=CSV_SEARCH_COLUMN,
    return_column=CSV_RETURN_COLUMN,
    idx_to_single_label_path=IDX_TO_SINGLE_LABEL_JSON,
):
    class_to_idx, idx_to_class = load_class_map(class_map_path)
    idx_to_short = load_idx_to_single_label(idx_to_single_label_path)
    model, device = load_model(model_weights_path, num_classes=len(class_to_idx))
    allowed_idx = load_allowed_indices(allowed_json_path, class_to_idx, idx_to_class)
    if allowed_idx:
        print(f"Allowed indices: {len(allowed_idx)} / {len(class_to_idx)}")

    df = None
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="latin1")
        if search_column not in df.columns or return_column not in df.columns:
            raise ValueError(f"CSV missing '{search_column}' or '{return_column}'")

    # only within this base_dir
    for root, _, _ in os.walk(base_dir):
        if os.path.basename(root) != "car_captures":
            continue
        video_dir = os.path.dirname(root)
        json_path = os.path.join(video_dir, "crossing_data.json")
        if not os.path.exists(json_path):
            print(f"[!] Skipping (no JSON): {video_dir}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            crossing_data = json.load(f)

        image_files = sorted([p for p in glob(os.path.join(root, "*"))
                              if p.lower().endswith((".jpg",".jpeg",".png"))])

        for i, image_path in enumerate(image_files):
            if i >= len(crossing_data):
                print(f"[!] Extra images in {root}, skipping from {i} onward.")
                break

            idx = predict_image(image_path, model, device,
                                allowed_indices=allowed_idx if allowed_idx else None)
            full_name = idx_to_class[idx]
            short_name = idx_to_short.get(idx, None)

            crossing_data[i]["predicted_class_idx"]   = idx
            crossing_data[i]["predicted_class_name"]  = full_name
            crossing_data[i]["predicted_label_short"] = short_name

            emission = find_emissions_by_short_label(short_name, df, search_column, return_column) \
                       if df is not None else None
            crossing_data[i]["Gas_Emission"] = 0 if emission is None else emission
            crossing_data[i]["Gas_Emission_lookup_key"] = short_name

            print(f"Predicted [{idx}] {full_name} (-> {short_name}) for {image_path}")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(crossing_data, f, ensure_ascii=False, indent=4)
        print(f"Updated: {json_path}")

# =========================
#  Pairs (single-folder)
# =========================
def create_pairs_for_single_video(video_output_dir):
    video_dir = Path(video_output_dir)
    car_dir = video_dir / "car_captures"
    audio_dir = video_dir / "audio_segments"
    pairs_dir = video_dir / "pairs"

    if not car_dir.exists():
        print(f"No car captures found in {video_dir.name}")
        return

    pairs_dir.mkdir(exist_ok=True)
    car_files = sorted(car_dir.glob("*.jpg"))
    audio_files = sorted(audio_dir.glob("*.wav"))
    for i, (car_file, audio_file) in enumerate(zip(car_files, audio_files)):
        folder_name = f"pair_{i+1}"
        pair_folder = pairs_dir / folder_name
        pair_folder.mkdir(exist_ok=True)
        shutil.copy2(car_file,  pair_folder / car_file.name)
        shutil.copy2(audio_file, pair_folder / audio_file.name)
        print(f"Created {folder_name} with {car_file.name} and {audio_file.name}")
    print(f"Created {min(len(car_files), len(audio_files))} pairs in {pairs_dir}")

# =========================
#   Orchestrate ONE video
# =========================
def process_single_video(input_video_path):
    """Full pipeline for exactly one video, then move source into its folder."""
    os.makedirs(OUTPUT_BASE_DIRECTORY, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    video_output_dir   = os.path.join(OUTPUT_BASE_DIRECTORY, video_name)
    car_captures_dir   = os.path.join(video_output_dir, "car_captures")
    audio_segments_dir = os.path.join(video_output_dir, "audio_segments")
    os.makedirs(car_captures_dir, exist_ok=True)
    os.makedirs(audio_segments_dir, exist_ok=True)

    # get width for lane lines
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[!] Cannot open {input_video_path}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    left_line_x  = frame_width // 4
    right_line_x = 2 * frame_width // 4

    output_video_path = os.path.join(video_output_dir, f"processed_{os.path.basename(input_video_path)}")

    # main processing
    car_images, crossing_durations = process_video_with_car_capture(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        left_line_x=left_line_x,
        right_line_x=right_line_x,
        output_image_dir=car_captures_dir,
        model_name='yolov8n.pt',
        detection_interval=1,
        min_confidence=0.6,
        min_car_width=100,
        min_car_height=100,
        road_distance=6
    )

    # crossing JSON
    json_output_path = os.path.join(video_output_dir, "crossing_data.json")
    with open(json_output_path, "w") as json_file:
        json.dump(crossing_durations, json_file, indent=4)
    print(f"Saved crossing data to {json_output_path}")

    # audio segments
    if crossing_durations and crossing_durations[0].get('start_time') is not None:
        temp_audio_path = os.path.join(video_output_dir, "temp_audio.wav")
        if extract_audio(input_video_path, temp_audio_path):
            detection_times = [ts['start_time'] for ts in crossing_durations]
            extract_segments(temp_audio_path, detection_times, audio_segments_dir, video_name, range_sec=1.0)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        print(f"Processed {len(cetection_times) if crossing_durations else 0} audio segments")
    else:
        print("No cars detected, skipping audio extraction")

    # pairs (only this folder)
    create_pairs_for_single_video(video_output_dir)

    # predictions (only this folder)
    predict_all_car_captures_and_GasEmissions(
        base_dir=video_output_dir,
        model_weights_path=PRED_MODEL_WEIGHTS,
        class_map_path=CLASS_MAP_JSON,
        allowed_json_path=ALLOWED_JSON,
        csv_path=CSV_PATH,
        search_column=CSV_SEARCH_COLUMN,
        return_column=CSV_RETURN_COLUMN,
        idx_to_single_label_path=IDX_TO_SINGLE_LABEL_JSON
    )

    # move source video into the processed folder for archiving
    dst_source = os.path.join(video_output_dir, "source.mp4")
    try:
        if os.path.abspath(input_video_path) != os.path.abspath(dst_source):
            shutil.move(input_video_path, dst_source)
            print(f"Moved source video to {dst_source}")
    except Exception as e:
        print(f"[!] Could not move source video: {e}")

# =========================
#   Button logic (toggle)
# =========================
isrec = False

def handle():
    global isrec
    while isrec:
        if detect_loud_sound():
            # 1) record
            path = record_one_clip()
            # 2) process that single file
            process_single_video(path)
        time.sleep(0.2)  # avoid busy loop

def press():
    global isrec
    isrec = not isrec
    print("rec" if isrec else "stop")
    set_led(1, 0, 1)  # magenta
    if isrec:
        handle()

def buttonrec():
    # startup blink
    set_led(1,1,0); time.sleep(0.4)
    set_led(1,0,1)
    button.when_pressed = press
    print("[READY] Press button to toggle listening for loud sound...")
    while True:
        time.sleep(1)

# =========================
#           MAIN
# =========================
set_led(1, 1, 0)
time.sleep(0.4)
set_led(1,1,1)

buttonrec()
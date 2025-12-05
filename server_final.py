import os
import io
import json
import threading
import time
from datetime import datetime, date, timedelta
from threading import Lock


from flask import Flask, request, jsonify, Response, send_from_directory, render_template_string
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
MANIFEST_FILE = "photos_manifest.json"
THUMBS_UP_STATUS = 1
THUMBS_DOWN_STATUS = 0
THUMBS_UNKNOWN_STATUS = 2

SERVO_CONTROL_IP = "172.20.10.12"
SERVO_CONTROL_PORT = 80

from zoneinfo import ZoneInfo
load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "London"
TIMEZONE = ZoneInfo("Europe/London")
PHOTO_ROOT = os.path.join("static", "photos")
WEATHER_ROOT = "weather"
MANIFEST = "photos_manifest.json"
LAST_IMG = os.path.join("static", "last.jpg")
PREV_IMG = os.path.join("static", "prev.jpg")
YS_THRESHOLD_FEELS = 18.0

os.makedirs(PHOTO_ROOT, exist_ok=True)
os.makedirs(WEATHER_ROOT, exist_ok=True)
os.makedirs("static", exist_ok=True)

app = Flask(__name__, static_folder="static")
lock = Lock()

model = YOLO("D:/ic/SIOT/clothes_identify_new/model_final/four_clothes_types_model3/weights/best.pt")
labels = ["thick_inner", "thin_inner", "down_jacket", "coat"]
clothes_warmth = {
    "thick_inner": 5,
    "thin_inner": 3,
    "down_jacket": 11,
    "coat": 9,
}

latest_detections = []
previous_detections = []

latest_image_path = LAST_IMG
previous_image_path = PREV_IMG

def is_warm_enough(cloth_name, feels_like_temp):
    base_score = clothes_warmth.get(cloth_name, 0)

    if feels_like_temp < 0:
        required_score = 11
    elif feels_like_temp <= 5:
        required_score = 9
    elif feels_like_temp <= 10:
        required_score = 7
    else:
        required_score = 5

    return base_score >= required_score

def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Unable to load Manifest file ({MANIFEST_FILE})，the file may be corrupted: {e}")
            return []
    return []

def save_manifest():
    try:
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] 无法保存 Manifest 文件: {e}")

# if os.path.exists(MANIFEST):
#     with open(MANIFEST, "r", encoding="utf-8") as f:
#         manifest = json.load(f)
# else:
#     manifest = []  # newest first
#
# def save_manifest():
#     with lock:
#         try:
#             with open(MANIFEST, "w", encoding="utf-8") as f:
#                 json.dump(manifest, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             print(f"[ERROR] Failed to save manifest: {e}")


def london_now():
    return datetime.now(TIMEZONE)

def date_str_from_dt(dt):
    return dt.strftime("%Y-%m-%d")

def timestamp_str_from_dt(dt):
    return dt.strftime("%Y%m%d_%H%M%S")

def ensure_day_folder(date_str):
    folder = os.path.join(PHOTO_ROOT, date_str)
    os.makedirs(folder, exist_ok=True)
    return folder

def fetch_current_weather():
    if not WEATHER_API_KEY:
        return {"error": "No WEATHER_API_KEY configured"}
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={CITY}&appid={WEATHER_API_KEY}&units=metric"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception as e:
        return {"error": f"request failed: {e}", "snapshot": None}
    if r.status_code != 200:
        return {"error": f"API returned error {r.status_code}: {data}", "snapshot": None}
    snapshot = {
        "temp": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "weather": data["weather"][0]["description"]
    }
    return {"error": None, "snapshot": snapshot}


def save_weather_for_date(dt):
    info = fetch_current_weather()
    if "error" in info:
        print(f"[WARNING] Weather fetch error: {info['error']}")
        return None

    dstr = date_str_from_dt(dt)
    path = os.path.join(WEATHER_ROOT, f"{dstr}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved weather cache for {dstr}")
    return info

def save_weather_for_today():
    save_weather_for_date(london_now())

# def get_weather_for_photo(photo_id):
#     path = os.path.join(WEATHER_ROOT, f"{photo_id}.json")
#     if not os.path.exists(path):
#         return None
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

def get_weather_for_date_str(date_str):
    path = os.path.join(WEATHER_ROOT, f"{date_str}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load weather cache for {date_str}: {e}")
        return None
# scheduler = BackgroundScheduler()
# scheduler.add_job(save_weather_for_today, 'interval', minutes=30)
# scheduler.start()
try:
    print("Pre-fetching weather data...")
    save_weather_for_today()
except Exception as e:
    print(f"Startup weather fetch failed: {e}")

def run_yolo_and_annotate(pil_img):
    results = model(pil_img)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append(label)
    annotated = results[0].plot()
    return detections, annotated

def save_image_files(pil_img, annotated_bgr, dt):
    dstr = date_str_from_dt(dt)
    ts = timestamp_str_from_dt(dt)
    folder = ensure_day_folder(dstr)
    orig_name = f"{ts}.jpg"
    ann_name = f"{ts}_annot.jpg"
    orig_path = os.path.join(folder, orig_name)
    ann_path = os.path.join(folder, ann_name)
    # save original
    pil_img.save(orig_path, format="JPEG", quality=85)
    # save annotated (BGR)
    cv2.imwrite(ann_path, annotated_bgr)
    # return relative web paths under static/
    rel_orig = os.path.join("photos", dstr, orig_name).replace("\\", "/")
    rel_ann = os.path.join("photos", dstr, ann_name).replace("\\", "/")
    return rel_orig, rel_ann, orig_path, ann_path

def compute_photo_metadata(detections, weather_snapshot):
    # per-item warm bools, and overall sum score
    feels = None
    if isinstance(weather_snapshot, dict) and "feels_like" in weather_snapshot:
        feels = weather_snapshot["feels_like"]
    item_results = []
    total_score = 0
    any_warm = False
    for lab in detections:
        base = clothes_warmth.get(lab, 0)
        total_score += base
        is_warm = False
        if feels is not None:
            is_warm = is_warm_enough(lab, feels)
            if is_warm:
                any_warm = True
        item_results.append({"label": lab, "score": base, "is_warm_enough": is_warm})
    overall = {
        "items": item_results,
        "warmth_score": total_score,
        "any_item_warm_enough": any_warm,
        "photo_level_warm_enough": any_warm if feels is not None else None
    }
    return overall

def add_photo_record(dt, rel_orig, rel_ann, detections, weather_snapshot, meta):
    ts = timestamp_str_from_dt(dt)
    rec = {
        "timestamp": ts,
        "datetime": dt.astimezone(TIMEZONE).isoformat(),
        "date": date_str_from_dt(dt),
        "orig": rel_orig,
        "annotated": rel_ann,
        "detections": detections,
        "weather": weather_snapshot,
        "warmth_meta": meta
    }
    with lock:
        manifest.insert(0, rec)  # newest first
        save_manifest()
    # also save per-photo json in same folder
    folder = ensure_day_folder(rec["date"])
    meta_path = os.path.join(folder, f"{ts}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return rec

def send_servo_command(status_code):
    url = f"http://{SERVO_CONTROL_IP}:{SERVO_CONTROL_PORT}/servo?status={status_code}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            print(f"[SERVO] Command sent successfully: {status_code}")
        else:
            print(f"[SERVO] Command sent unsuccessfully: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[SERVO ERROR]: {e}")

def background_processing(pil_img, detections, annotated_bgr):
    global lock, latest_detections, previous_detections

    start_time = time.time()
    print("--- [BACKGROUND] 1. Starting... ")

    try:
        dt = london_now()
        date_str = date_str_from_dt(dt)
        if os.path.exists(LAST_IMG):
            try:
                if os.path.exists(PREV_IMG):
                    os.remove(PREV_IMG)
                os.rename(LAST_IMG, PREV_IMG)
            except Exception as e:
                print(f"[WARNING] Failed to move file: {e}")

        cv2.imwrite(LAST_IMG, annotated_bgr)
        print(f"[DEBUG] Img has been saved: {time.time() - start_time:.2f}s")

        with lock:
            previous_detections = latest_detections.copy() if latest_detections else []
            latest_detections = detections

        print(f"--- [BACKGROUND] 2. Key update finished")

    except Exception as e:
        print(f"[CRITICAL ERROR] Stage 1 失败: {e}")
    try:
        weather_result = fetch_current_weather()
        weather_snapshot = weather_result["snapshot"]

        if weather_result["error"]:
            print(f"[WARNING] Failed to get real-time weather: {weather_result['error']}")
            weather_snapshot = {"temp": "N/A", "feels_like": "N/A", "weather": "Unknown (API Error)"}
        rel_orig, rel_ann, orig_path, ann_path = save_image_files(pil_img, annotated_bgr, dt)
        meta = compute_photo_metadata(detections, weather_snapshot)

        rec = add_photo_record(dt, rel_orig, rel_ann, detections, weather_snapshot, meta)

        print(f"--- [BACKGROUND] 3. History record has been saved: {time.time() - start_time:.2f}s")
        final_warmth_status = meta["photo_level_warm_enough"]

        if final_warmth_status is True:
            send_servo_command(THUMBS_UP_STATUS)
        elif final_warmth_status is False:
            send_servo_command(THUMBS_DOWN_STATUS)
        else:
            send_servo_command(THUMBS_UNKNOWN_STATUS)

    except Exception as e:
        print(f"[CRITICAL ERROR] Stage 2 失败: {e}")

@app.route("/")
def home():
    return render_template_string("""
      <html>
        <head><meta charset="utf-8"><title>Smart Closet</title></head>
        <body>
          <h2>Smart Closet Server</h2>
          <p><a href="/view">Open live view</a></p>
        </body>
      </html>
    """)

@app.route("/upload", methods=["POST"])
def upload():
    print("------------------------------------------------")
    print("1. [Server] Upload request received! Receiving data...")
    global latest_detections, previous_detections
    img_bytes = request.data
    if not img_bytes:
        print("   [Error] No data received")
        return jsonify({"error": "No data"}), 400
    print(f"2. [Server] Data reception complete,size: {len(img_bytes)} bytes")

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        print("3. [Server] Image decoded successfully")
    except Exception as e:
        print(f"   [Error] Image decoded unsuccessfully: {e}")
        return jsonify({"error": f"Image decode failed: {e}"}), 400

    print("4. [Server] Ready to run YOLO...")
    detections, annotated_bgr = run_yolo_and_annotate(pil_img)
    print("5. [Server] YOLO stage finished, saving file...")
    thread_args = (pil_img, detections, annotated_bgr)
    thread = threading.Thread(target=background_processing, args=thread_args)
    thread.start()
    print("Background thread started. Sending fast response now.")

    return jsonify({
        "message": "processing_started",
        "detections_count": len(detections),
        "status": "Check the view page in a moment"
    }), 200

# def save_weather_for_photo(dt, photo_id):
#     start = time.time()
#     info = fetch_current_weather()
#     end_fetch = time.time()
#     print(f"[Weather] API call took {end_fetch - start:.2f}s")
#
#     if "error" in info:
#         print(f"[WARNING] Weather fetch error: {info['error']}")
#         return None
#
#     if not os.path.exists(WEATHER_ROOT):
#         os.makedirs(WEATHER_ROOT)
#
#     filename = f"{photo_id}.json"
#     path = os.path.join(WEATHER_ROOT, filename)
#
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(info, f, ensure_ascii=False, indent=2)
#
#     end_total = time. time()
#     print(f"[Weather] Save to file took {end_total - end_fetch:.2f}s, total {end_total - start:.2f}s")
#     return info

@app.route("/latest_image")
def latest_image():
    if not os.path.exists(LAST_IMG):
        return jsonify({"error": "no latest image"}), 404
    return send_from_directory("static", "last.jpg")

@app.route("/latest_text")
def latest_text():
    return jsonify({"detections": latest_detections, "previous": previous_detections})

@app.route("/weather")
def weather_today():
    result = fetch_current_weather()
    if result["error"]:
        return jsonify(result), 500

    return jsonify(result["snapshot"])
    # date_str = date_str_from_dt(london_now())
    # w = get_weather_for_date_str(date_str) or fetch_current_weather()
    # return jsonify(w)
# @app.route("/weather")
# def weather_for_photo():
#
#     photo_id = request.args.get("photo_id")
#     if photo_id:
#         path = os.path.join(WEATHER_ROOT, f"{photo_id}.json")
#         if os.path.exists(path):
#             with open(path, "r", encoding="utf-8") as f:
#                 w = json.load(f)
#             return jsonify(w)
#         else:
#             return jsonify({"error": "Photo weather not found"}), 404
#     else:
#         all_files = sorted(os.listdir(WEATHER_ROOT), reverse=True)
#         if all_files:
#             latest_file = os.path.join(WEATHER_ROOT, all_files[0])
#             with open(latest_file, "r", encoding="utf-8") as f:
#                 w = json.load(f)
#             return jsonify(w)
#         else:
#             return jsonify({"error": "No weather data available"}), 404
@app.route('/api/stats')
def get_stats_data():
    with lock:
        sorted_data = sorted(manifest, key=lambda x: x['timestamp'])

    timestamps = []
    temps = []
    scores = []
    statuses = []
    status_counts = {"Cold": 0, "Warm Enough": 0}

    for item in sorted_data:
        try:
            dt_str = item.get("datetime", "")
            if dt_str:
                dt_obj = datetime.fromisoformat(dt_str)
                t_str = dt_obj.strftime("%m-%d %H:%M")
            else:
                t_str = item.get("timestamp", "N/A")

            weather = item.get("weather", {})
            if isinstance(weather, dict):
                feels_like = weather.get("feels_like", 0)
            else:
                feels_like = 0

            meta = item.get("warmth_meta", {})
            score = meta.get("warmth_score", 0)

            if feels_like < 0:
                req = 11
            elif feels_like <= 5:
                req = 9
            elif feels_like <= 10:
                req = 7
            else:
                req = 5

            status_code = 1
            label = "Warm Enough"

            if score < req:
                status_code = 0
                label = "Cold"

            timestamps.append(t_str)
            temps.append(feels_like)
            scores.append(score)
            statuses.append(status_code)
            status_counts[label] += 1

        except Exception as e:
            print(f"[Dashboard Error] Skipping item: {e}")
            continue

    return jsonify({
        "timestamps": timestamps,
        "temps": temps,
        "scores": scores,
        "statuses": statuses,
        "counts": status_counts
    })

@app.route("/photos")
def photos_by_date():
    date_str = request.args.get("date", date_str_from_dt(london_now()))
    with lock:
        filtered = [m for m in manifest if m["date"] == date_str]
    return jsonify(filtered)

@app.route("/photo/<path:filename>")
def serve_photo(filename):
    safe_root = os.path.abspath("static")
    target = os.path.abspath(os.path.join("static", filename))
    if not target.startswith(safe_root) or not os.path.exists(target):
        return jsonify({"error": "file not found"}), 404
    reldir = os.path.dirname(filename)
    fname = os.path.basename(filename)
    return send_from_directory(os.path.join("static", reldir), fname)

@app.route("/dashboard")
def dashboard_page():
    return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Smart Closet Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }

    /* Navigation bar style */
    .nav { margin-bottom: 20px; background: #333; padding: 10px; border-radius: 5px; }
    .nav a { color: white; text-decoration: none; margin-right: 20px; font-weight: bold; font-size: 16px; }
    .nav a.active { color: #4CAF50; }

    /* Chart Container */
    .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .chart-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .full-width { grid-column: 1 / -1; }

    h2 { text-align: center; color: #333; }
  </style>
</head>
<body>

  <div class="nav">
    <a href="/view">📷 Live View & History</a>
    <a href="/dashboard" class="active">📊 Data Dashboard</a>
  </div>

  <h2>Weekly Analysis (Based on Real Data)</h2>

  <div class="charts-grid">
    <div class="chart-card full-width">
      <canvas id="tempWarmthChart"></canvas>
    </div>

    <div class="chart-card">
      <canvas id="habitPieChart"></canvas>
    </div>

    <div class="chart-card">
      <canvas id="statusTrendChart"></canvas>
    </div>
  </div>

<script>
  async function loadDashboard() {
    // 1. Obtain real data
    const res = await fetch('/api/stats');
    const data = await res.json();

    if (data.timestamps.length === 0) {
        alert("Not enough data to generate a chart. Please upload some photos first!");
        return;
    }

    // --- Chart 1: Temperature (Line) vs Warmth (Bar) ---
    const ctx1 = document.getElementById('tempWarmthChart').getContext('2d');
    new Chart(ctx1, {
        type: 'bar',
        data: {
            labels: data.timestamps,
            datasets: [
                {
                    label: 'Warmth Score (Bar)',
                    data: data.scores,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    yAxisID: 'y'
                },
                {
                    label: 'Feels Like Temp °C (Line)',
                    data: data.temps,
                    type: 'line',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'Temperature vs. Outfit Warmth' } },
            scales: {
                y: { type: 'linear', display: true, position: 'left', title: {display:true, text:'Score'} },
                y1: { type: 'linear', display: true, position: 'right', title: {display:true, text:'Temp (°C)'} }
            }
        }
    });

    // --- Chart 2: Dressing Habit (Pie) ---
    const ctx2 = document.getElementById('habitPieChart').getContext('2d');
    
    // Extract data for Cold and Warm Enough
    const labels2 = ['Cold (Underdressed)', 'Warm Enough (Suitable)'];
    const counts2 = [data.counts.Cold || 0, data.counts["Warm Enough"] || 0];
    const backgroundColors2 = ['#FF6384', '#4CAF50']; 

    new Chart(ctx2, {
        type: 'pie',
        data: {
            labels: labels2,
            datasets: [{
                data: counts2,
                backgroundColor: backgroundColors2
            }]
        },
        options: {
            plugins: { 
                title: { display: true, text: 'Dressing Habits Distribution (Warm Enough / Cold)' } 
            }
        }
    });

    // --- Chart 3: Comfort (Line Step) ---
    const ctx3 = document.getElementById('statusTrendChart').getContext('2d');
    new Chart(ctx3, {
        type: 'line',
        data: {
            labels: data.timestamps,
            datasets: [{
                label: 'Status (0:Cold, 1:Warm Enough)',
                data: data.statuses,
                borderColor: '#007bff', 
                backgroundColor: '#007bff',
                stepped: true, 
                borderWidth: 2
            }]
        },
        options: {
            scales: {
                y: { 
                    min: 0, max: 1, 
                    ticks: { 
                        stepSize: 1,
                        callback: function(val) { 
                            return ['Cold (Underdressed)', 'Warm Enough'][val]; 
                        } 
                    } 
                }
            },
            plugins: { title: { display: true, text: 'Comfort Judgment Trend' } }
        }
    });
  }

  loadDashboard();
</script>
</body>
</html>
    """)

@app.route("/view")
def view_page():
    return render_template_string("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Smart Closet Live</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }

    /* Navigation bar style */
    .nav { margin-bottom: 20px; background: #333; padding: 10px; border-radius: 5px; }
    .nav a { color: white; text-decoration: none; margin-right: 20px; font-weight: bold; font-size: 16px; }
    .nav a.active { color: #4CAF50; }

    #cam { border: 1px solid #ddd; border-radius: 8px; }
    .photo-card { border:1px solid #eee; padding:10px; margin:10px 0; display:flex; gap:15px; align-items:center; background:white; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
    .thumb { width:320px; border-radius:4px; }
    .meta { font-size:14px; line-height: 1.6; }
    #photos { margin-top:20px; }
  </style>
</head>
<body>

  <div class="nav">
    <a href="/view" class="active">📷 Live View & History</a>
    <a href="/dashboard">📊 Data Dashboard</a>
  </div>

  <h2>📷 Smart Closet — Live</h2>

  <div>
    <img id="cam" src="/latest_image" width="640" alt="live">
    <div id="det" style="margin-top:10px; font-size:18px;"><b>Detections:</b> Loading...</div>
    <div id="todayWeather" style="margin-top:5px; color:#555;"></div>
  </div>

  <hr>

  <h3>History (Select Date)</h3>
  <label for="date_select">Date: </label>
  <input type="date" id="date_select" />

  <div id="photos">Loading photos...</div>

<script>
  const dateInput = document.getElementById("date_select");
  const today = new Date().toISOString().split('T')[0];
  dateInput.value = today;

  function refreshLive() {
    document.getElementById("cam").src = "/latest_image?t=" + Date.now();

    fetch("/latest_text").then(r=>r.json()).then(data=>{
      document.getElementById("det").innerHTML = "<b>Detections:</b> " + (data.detections.join(", ") || "None");
    });

    fetch("/weather").then(r=>r.json()).then(w=>{
      if (w.error) {
        document.getElementById("todayWeather").innerText = "Weather error: " + JSON.stringify(w);
      } else {
        document.getElementById("todayWeather").innerHTML = `<b>Weather (Realtime):</b> ${w.weather}, Temp ${w.temp}°C, Feels ${w.feels_like}°C`;
      }
    });
  }

  async function loadPhotosForDate(d) {
    document.getElementById("photos").innerText = "Loading...";
    try {
        const res = await fetch("/photos?date=" + d);
        if (!res.ok) {
        document.getElementById("photos").innerText = "Server Error: " + res.status;
        return;
        }
        const list = await res.json();
        if (!list || list.length === 0) {
        document.getElementById("photos").innerText = "No photos for " + d;
        return;
        }
        let html = "";
        for (const item of list) {
        const ann = item.annotated || "";
        const dt = item.timestamp;
        const dets = Array.isArray(item.detections) ? item.detections.join(", ") : "None";

        let weather = "N/A";
        let feels = "N/A";
        if (item.weather) {
            weather = item.weather.weather || "N/A";
            feels = item.weather.feels_like !== undefined ? item.weather.feels_like : "N/A";
        }

        const warmth = item.warmth_meta || {};
        const warmth_score = warmth.warmth_score !== undefined ? warmth.warmth_score : 0;

        let statusText = 'Unknown';
        if (warmth.photo_level_warm_enough === true) statusText = '<span style="color:green">Suitable</span>';
        else if (warmth.photo_level_warm_enough === false) statusText = '<span style="color:red">Underdressed</span>';

        html += `<div class="photo-card">
                    <div><img class="thumb" src="/static/${ann}" onerror="this.src=''"></div>
                    <div class="meta">
                    <div><b>${dt}</b></div>
                    <div>🎯 Detections: ${dets}</div>
                    <div>🌡️ Weather: ${weather}, Feels ${feels}°C</div>
                    <div>🔥 Score: ${warmth_score}</div>
                    <div>🤖 Judgment: ${statusText}</div>
                    </div>
                </div>`;
        }
        document.getElementById("photos").innerHTML = html;
    } catch (err) {
        console.error(err);
        document.getElementById("photos").innerText = "JS Error";
    }
  }

  refreshLive();
  setInterval(refreshLive, 2000); 
  setInterval(()=> loadPhotosForDate(dateInput.value), 5000);

  dateInput.addEventListener("change", ()=> loadPhotosForDate(dateInput.value));
  loadPhotosForDate(dateInput.value);
</script>

</body>
</html>
    """)
# --------------------
# Shutdown scheduler gracefully
# --------------------

# import atexit
# @atexit.register
# def shutdown_scheduler():
#     try:
#         scheduler.shutdown()
#     except Exception:
#         pass
manifest = load_manifest()
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

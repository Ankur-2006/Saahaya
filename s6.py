"""
Sahaaya v3.0 — Women Safety Platform
Professional UI • Photo Upload • Auto Address • Live Map • Safety Gauge • Video Progress • Panic (SMS+Call)

Environment vars (recommended in .env)
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_FROM
- POLICE_NUMBER
- SIREN_MP3_URL (default uses a public alarm sound)

Run
pip install flask flask_sqlalchemy twilio opencv-python-headless numpy deepface
python s1.py
"""

import os
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from deepface import DeepFace
from twilio.rest import Client

# ---------------- CONFIG ----------------
DATABASE_URL = "sqlite:///safety_app.db"
VIDEO_FILE_PATH = "walking.mp4"  # local demo video
UPLOAD_FOLDER = "photos"
ALLOWED_IMAGE_EXTS = {"png","jpg","jpeg"}

# Environment-safe secrets
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN",  "")
TWILIO_FROM        = os.getenv("TWILIO_FROM",        "")
POLICE_NUMBER      = os.getenv("POLICE_NUMBER",       None)
SIREN_MP3_URL      = os.getenv("SIREN_MP3_URL",       "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg")

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB photo
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db = SQLAlchemy(app)

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# In-memory progress/state per user
PROGRESS = {}

# ---------------- MODELS ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    phone = db.Column(db.String(30), unique=True, nullable=False)
    email = db.Column(db.String(120))
    address = db.Column(db.String(300))
    photo_path = db.Column(db.String(300))
    trusted_name = db.Column(db.String(120))
    trusted_phone = db.Column(db.String(30))

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    message = db.Column(db.String(500))
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)

@app.before_request
def create_tables_once():
    if not hasattr(app, "_tables_created"):
        with app.app_context():
            db.create_all()
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        app._tables_created = True

# ---------------- TWILIO HELPERS ----------------
def send_sms(to, body):
    if not to:
        return False
    try:
        twilio_client.messages.create(body=body, from_=TWILIO_FROM, to=to)
        print(f"SMS sent to {to}")
        return True
    except Exception as e:
        print(f"SMS error to {to}: {e}")
        return False

def make_call_with_siren(to):
    if not to:
        return False
    try:
        twilio_client.calls.create(
            to=to,
            from_=TWILIO_FROM,
            twiml=f"<Response><Play>{SIREN_MP3_URL}</Play></Response>"
        )
        print(f"Call placed to {to}")
        return True
    except Exception as e:
        print(f"Call error to {to}: {e}")
        return False

# ---------------- VISION / SCORING ----------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def count_people(frame):
    if frame is None:
        return 0
    resized = cv2.resize(frame, (640, 480))
    rects, _ = hog.detectMultiScale(resized, winStride=(8, 8))
    return len(rects)


def analyze_faces(frame):
    results = []
    try:
        faces = DeepFace.extract_faces(img_path=frame, detector_backend="opencv", enforce_detection=False)
        for f in faces:
            face_img = f.get("face")
            if face_img is None:
                continue
            analysis = DeepFace.analyze(face_img, actions=["emotion", "age", "gender"], enforce_detection=False)
            results.append({
                "emotion": (analysis.get("dominant_emotion") or "").lower(),
                "gender": (analysis.get("gender") or "").lower()
            })
    except Exception as e:
        print(f"DeepFace warn: {e}")
    return results


def compute_safety_score(num_people, male_count, female_count, angry, sad):
    score = 100
    if num_people > 20:
        score -= 30
    elif num_people > 8:
        score -= 10
    if male_count + female_count > 0:
        male_ratio = male_count / max(1, (male_count + female_count))
        score -= int(30 * male_ratio)
    score -= int(5 * angry)
    score -= int(3 * sad)
    return max(0, min(100, score))


def heuristic_location_adjust(base_score):
    hour = datetime.now().hour
    adj = 0
    if hour < 6 or hour >= 22:
        adj -= 20
    elif hour < 8 or hour >= 20:
        adj -= 10
    return max(0, min(100, base_score + adj))

# ---------------- VIDEO PROCESSING (with progress) ----------------
def process_video(user_id: int, video_path: str):
    user = User.query.get(user_id)
    if not user or not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    PROGRESS[user_id] = {"total": total, "done": 0, "status": "running", "metrics": {}, "last_score": PROGRESS.get(user_id, {}).get("last_score", 75)}

    total_people = male_count = female_count = angry = sad = 0
    frame_idx = 0
    sample_rate = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            PROGRESS[user_id]["done"] = min(total, frame_idx)
            continue

        total_people += count_people(frame)
        faces = analyze_faces(frame)
        for f in faces:
            if "angry" in f.get("emotion", ""): angry += 1
            if "sad" in f.get("emotion", ""): sad += 1
            g = f.get("gender", "")
            if "male" in g: male_count += 1
            elif "female" in g: female_count += 1

        PROGRESS[user_id]["done"] = min(total, frame_idx)

    cap.release()

    score = compute_safety_score(total_people, male_count, female_count, angry, sad)
    score = heuristic_location_adjust(score)

    PROGRESS[user_id]["metrics"] = {"people": total_people, "male": male_count, "female": female_count, "angry": angry, "sad": sad, "score": score}
    PROGRESS[user_id]["last_score"] = score
    PROGRESS[user_id]["status"] = "done"

    if score < 40 or angry > 2:
        msg = f"ALERT: Safety score {score}. Possible danger detected for {user.name}!"
        db.session.add(Alert(user_id=user.id, message=msg))
        db.session.commit()
        send_sms(user.phone, msg)
        if user.trusted_phone:
            send_sms(user.trusted_phone, f"{msg} (Trusted Contact)")
        if POLICE_NUMBER:
            send_sms(POLICE_NUMBER, f"{msg} (Police)")

# ---------------- HELPERS ----------------
def allowed_image(filename:str)->bool:
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_IMAGE_EXTS

# ---------------- ROUTES (UI) ----------------
@app.route("/", methods=["GET"])
def home():
    return render_template_string(FORM_HTML)

@app.route("/register", methods=["POST"])
def register():
    # multipart/form-data
    name = request.form.get("name")
    phone = request.form.get("phone")
    email = request.form.get("email")
    address = request.form.get("address")
    trusted_name = request.form.get("trusted_name")
    trusted_phone = request.form.get("trusted_phone")

    photo_path = None
    file = request.files.get("photo")
    if file and allowed_image(file.filename):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        ext = file.filename.rsplit(".",1)[1].lower()
        filename = f"photo_{int(datetime.utcnow().timestamp())}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        photo_path = save_path

    user = User(name=name, phone=phone, email=email, address=address, photo_path=photo_path,
                trusted_name=trusted_name, trusted_phone=trusted_phone)
    db.session.add(user)
    db.session.commit()

    PROGRESS[user.id] = {"total":1, "done":0, "status":"idle", "metrics":{}, "last_score":75}

    return render_template_string(DASHBOARD_HTML, user=user)

# ---------------- ROUTES (API) ----------------
@app.route("/process_video", methods=["POST"])
def api_process_video():
    data = request.get_json(force=True)
    user_id = int(data.get("user_id"))
    PROGRESS[user_id] = {"total":1, "done":0, "status":"running", "metrics":{}, "last_score":PROGRESS.get(user_id,{}).get("last_score",75)}
    threading.Thread(target=process_video, args=(user_id, VIDEO_FILE_PATH)).start()
    return jsonify({"ok": True})

@app.route("/video_status")
def video_status():
    user_id = int(request.args.get("user_id"))
    st = PROGRESS.get(user_id, {"total":1, "done":0, "status":"idle", "metrics":{}, "last_score":75})
    pct = 0
    if st["total"]:
        pct = int(100 * min(1.0, max(0.0, st["done"]/float(st["total"]))));
    return jsonify({"status": st.get("status","idle"), "progress": pct, "metrics": st.get("metrics", {}), "last_score": st.get("last_score",75)})

@app.route("/update_location", methods=["POST"])
def update_location():
    data = request.get_json(force=True)
    user_id = int(data.get("user_id"))
    lat = float(data.get("lat"))
    lng = float(data.get("lng"))
    st = PROGRESS.get(user_id, {})
    base = st.get("metrics",{}).get("score", st.get("last_score", 75))
    score = heuristic_location_adjust(base)
    st["last_score"] = score
    PROGRESS[user_id] = st
    return jsonify({"safety_score": score})

@app.route("/panic_alert", methods=["POST"])
def panic_alert():
    data = request.get_json(force=True)
    user_id = int(data.get("user_id"))
    user = User.query.get(user_id)
    if not user:
        return jsonify({"ok": False, "error": "user not found"}), 404

    msg = f"PANIC ALERT: {user.name} may be in danger. Phone: {user.phone}."
    # SMS to trusted + police
    sent = {
        "trusted_sms": send_sms(user.trusted_phone, msg),
        "police_sms": send_sms(POLICE_NUMBER, msg) if POLICE_NUMBER else False,
        "trusted_call": make_call_with_siren(user.trusted_phone),
        "police_call": make_call_with_siren(POLICE_NUMBER) if POLICE_NUMBER else False,
    }

    # Store alert
    db.session.add(Alert(user_id=user.id, message=msg))
    db.session.commit()

    return jsonify({"ok": True, "result": sent})

# (Optional) TwiML voice route if you prefer webhook style; using inline TwiML above so not required
@app.route("/twilio/voice", methods=["POST"])
def twilio_voice():
    return f"<Response><Play>{SIREN_MP3_URL}</Play></Response>", 200, {"Content-Type": "application/xml"}

@app.route("/alerts/<int:user_id>")
def alerts(user_id):
    rows = Alert.query.filter_by(user_id=user_id).order_by(Alert.sent_at.desc()).all()
    html = "<h2 style='font-family:Inter,Arial'>Alerts</h2>"
    if not rows:
        html += "<p>No alerts yet ✅</p>"
    for a in rows:
        html += f"<p>⚠️ {a.message}<br><small>{a.sent_at}</small></p>"
    html += "<p><a href='/' style='text-decoration:none'>&larr; Back</a></p>"
    return html

# ---------------- TEMPLATES ----------------
FORM_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sahaaya • Register</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap" rel="stylesheet">
  <style>
    body{font-family:Poppins,system-ui,Arial;background:linear-gradient(135deg,#f8bbd0,#fce4ec);min-height:100vh;display:flex;align-items:center;justify-content:center}
    .card{border:none;border-radius:22px;backdrop-filter: blur(6px);box-shadow:0 20px 60px rgba(233,30,99,.25);background:rgba(255,255,255,.85)}
    .brand{font-weight:800;color:#e91e63;letter-spacing:.5px}
    .form-control{border-radius:12px;padding:12px}
    .btn-brand{background:#e91e63;border:none;border-radius:12px;padding:12px 16px}
    .btn-brand:hover{background:#c2185b}
    .hint{font-size:12px;color:#666}
  </style>
</head>
<body>
  <div class="container">
    <div class="row justify-content-center">
      <div class="col-12 col-lg-8">
        <div class="card p-4 p-md-5">
          <h1 class="brand mb-1">🌸 Sahaaya</h1>
          <p class="text-muted mb-4">Register to start real-time safety insights.</p>
          <form method="POST" action="/register" enctype="multipart/form-data" class="row g-3">
            <div class="col-md-6">
              <label class="form-label">Name</label>
              <input name="name" class="form-control" required>
            </div>
            <div class="col-md-6">
              <label class="form-label">Phone</label>
              <input name="phone" class="form-control" required>
            </div>
            <div class="col-md-6">
              <label class="form-label">Email</label>
              <input name="email" type="email" class="form-control" required>
            </div>
            <div class="col-md-6">
              <label class="form-label">Trusted Contact Name</label>
              <input name="trusted_name" class="form-control" required>
            </div>
            <div class="col-md-6">
              <label class="form-label">Trusted Contact Phone</label>
              <input name="trusted_phone" class="form-control" required>
            </div>
            <div class="col-md-6">
              <label class="form-label">Profile Photo</label>
              <input name="photo" type="file" accept="image/*" class="form-control">
              <div class="hint">JPEG/PNG up to 8MB</div>
            </div>
            <div class="col-12">
              <label class="form-label">Current Address</label>
              <input id="address" name="address" class="form-control" placeholder="Fetching your location..." required>
              <div class="hint">Auto-filled from your location. Edit if incorrect.</div>
            </div>
            <div class="col-12 d-flex gap-2">
              <button type="button" class="btn btn-outline-secondary" onclick="autofillAddress()">📍 Use my location</button>
              <button class="btn btn-brand flex-grow-1">Create account</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  </div>

<script>
async function reverseGeocode(lat, lon){
  const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lon}`;
  const r = await fetch(url, {headers:{'Accept':'application/json'}});
  const d = await r.json();
  return d.display_name || `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
}
async function autofillAddress(){
  const inp = document.getElementById('address');
  if(!navigator.geolocation){ inp.placeholder = 'Geolocation not supported'; return; }
  navigator.geolocation.getCurrentPosition(async (pos)=>{
    const lat = pos.coords.latitude, lon = pos.coords.longitude;
    inp.value = 'Fetching address...';
    try{ inp.value = await reverseGeocode(lat, lon); }catch(e){ inp.value = `${lat.toFixed(5)}, ${lon.toFixed(5)}`; }
  }, ()=>{ inp.placeholder='Permission denied. Enter address manually.'; });
}
// auto-attempt on load
autofillAddress();
</script>
</body>
</html>
"""

DASHBOARD_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Sahaaya • Dashboard</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"/>
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <style>
    body{background:linear-gradient(180deg,#fff7fb,#ffffff);font-family:Poppins,system-ui,Arial}
    .appbar{background:#e91e63;color:white;padding:14px 20px;border-bottom-left-radius:16px;border-bottom-right-radius:16px;box-shadow:0 10px 30px rgba(233,30,99,.3)}
    .brand{font-weight:800}
    .pill{background:white;color:#e91e63;font-weight:600;border-radius:999px;padding:6px 12px}
    .gauge{width:220px;height:220px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(#e91e63 var(--p,0%),#ffe1ec 0);transition:background .4s}
    .gauge .inner{width:180px;height:180px;border-radius:50%;background:white;display:grid;place-items:center;box-shadow:inset 0 0 0 8px #ffe1ec}
    .gauge .value{font-size:42px;font-weight:800}
    .cardx{border:none;border-radius:20px;box-shadow:0 16px 40px rgba(0,0,0,.08);background:white}
    #map{height:300px;border-radius:16px}
    .analyze-box{height:300px;border-radius:16px;background:white;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px}
    .spinner{border:6px solid #f3f3f3;border-top:6px solid #e91e63;border-radius:50%;width:56px;height:56px;animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .btn-primary{background:#e91e63;border:none;border-radius:12px}
    .btn-primary:hover{background:#c2185b}
    .panic{position:fixed;right:24px;bottom:24px;background:#ff1744;border:none;color:white;font-weight:800;border-radius:999px;padding:16px 22px;box-shadow:0 12px 30px rgba(255,23,68,.45);transition:transform .1s}
    .panic:hover{transform:scale(1.03)}
    .avatar{width:56px;height:56px;border-radius:50%;object-fit:cover;border:3px solid #ffe1ec}
    .muted{color:#6b7280}
  </style>
</head>
<body>
  <div class=\"appbar d-flex align-items-center justify-content-between\">
    <div class=\"brand\">🌸 Sahaaya</div>
    <div class=\"pill\">Welcome, {{ user.name }}</div>
  </div>

  <div class=\"container py-4\">
    <div class=\"row g-4\">
      <div class=\"col-lg-4\">
        <div class=\"card cardx p-4 text-center\">
          <div class=\"gauge mx-auto mb-3\" id=\"gauge\" style=\"--p:75%\">
            <div class=\"inner\"><div class=\"value\" id=\"scoreValue\">75</div></div>
          </div>
          <div class=\"text-muted\">Live Safety Score</div>
          <button class=\"btn btn-primary mt-3\" id=\"startBtn\">▶ Start Video Analysis</button>
          <a class=\"d-block mt-2\" href=\"/alerts/{{ user.id }}\">View Alerts</a>
        </div>
      </div>

      <div class=\"col-lg-4\">
        <div class=\"card cardx p-3\"><div id=\"map\"></div></div>
      </div>

      <div class=\"col-lg-4\">
        <div class=\"card cardx p-3\">
          <div class=\"d-flex align-items-center gap-3\">
            {% if user.photo_path %}<img class=\"avatar\" src=\"/{{ user.photo_path }}\" alt=\"avatar\">{% else %}
            <div class=\"avatar d-flex align-items-center justify-content-center\">👤</div>{% endif %}
            <div>
              <div class=\"fw-bold\">{{ user.name }}</div>
              <div class=\"muted small\">{{ user.email }}</div>
            </div>
          </div>
          <hr>
          <div class=\"small muted\">Address</div>
          <div class=\"mb-2\">{{ user.address or '—' }}</div>
          <div class=\"small muted\">Trusted Contact</div>
          <div>{{ user.trusted_name }} — <span class=\"muted\">{{ user.trusted_phone }}</span></div>
        </div>
      </div>

      <div class=\"col-lg-12\">
        <div class=\"card cardx p-4 d-flex flex-row align-items-center justify-content-between\">
          <div class=\"d-flex align-items-center gap-3\">
            <div class=\"spinner\" id=\"spinner\"></div>
            <div>
              <div class=\"fw-semibold\" id=\"videoStatus\">Idle</div>
              <div class=\"small text-muted\">Video Progress: <span id=\"videoPct\">0</span>%</div>
            </div>
          </div>
          <button class=\"btn btn-primary\" id=\"startBtn2\">Analyze Again</button>
        </div>
      </div>
    </div>
  </div>

  <button class=\"panic\" onclick=\"panicNow()\">🚨 Panic</button>

<script>
  const userId = {{ user.id }};

  // Gauge helpers
  function setGauge(val){
    const gauge = document.getElementById('gauge');
    const value = document.getElementById('scoreValue');
    value.textContent = Math.round(val);
    gauge.style.setProperty('--p', Math.max(0, Math.min(100, val)) + '%');
  }

  // Start video analysis and poll status
  async function startAnalysis(){
    document.getElementById('videoStatus').textContent = 'Starting...';
    await fetch('/process_video', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({user_id: userId})});
  }
  document.getElementById('startBtn').addEventListener('click', startAnalysis);
  document.getElementById('startBtn2').addEventListener('click', startAnalysis);

  async function pollVideo(){
    try{
      const r = await fetch(`/video_status?user_id=${userId}`);
      const d = await r.json();
      document.getElementById('videoStatus').textContent = d.status === 'done' ? 'Completed' : (d.status === 'running' ? 'Analyzing...' : 'Idle');
      document.getElementById('videoPct').textContent = d.progress;
      if(d.metrics && d.metrics.score){ setGauge(d.metrics.score); }
    }catch(e){}
    setTimeout(pollVideo, 800);
  }
  pollVideo();

  // Live location → update safety score
  let map, marker;
  function initMap(){
    map = L.map('map');
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19}).addTo(map);
    map.setView([20.5937, 78.9629], 4); // India default
  }
  function updateMarker(lat,lng){
    if(!marker){ marker = L.marker([lat,lng]).addTo(map); }
    marker.setLatLng([lat,lng]);
    map.setView([lat,lng], 14);
  }
  initMap();

  if(navigator.geolocation){
    navigator.geolocation.watchPosition(async (pos)=>{
      const lat = pos.coords.latitude, lng = pos.coords.longitude;
      updateMarker(lat,lng);
      try{
        const r = await fetch('/update_location', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({user_id: userId, lat, lng})});
        const d = await r.json();
        if(typeof d.safety_score === 'number'){ setGauge(d.safety_score); }
      }catch(e){}
    });
  }

  // Panic button
  async function panicNow(){
    try{
      const r = await fetch('/panic_alert', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({user_id: userId})});
      const d = await r.json();
      alert(d.ok ? '🚨 Panic alert sent to trusted contact and police.' : 'Could not send alert.');
    }catch(e){ alert('Network error while sending panic alert'); }
  }
</script>
</body>
</html>
"""

# ---------------- STATIC FILE SERVE (photos) ----------------
@app.route('/photos/<path:filename>')
def static_photos(filename):
    # Allow serving uploaded photos
    return app.send_static_file(os.path.join('..', UPLOAD_FOLDER, filename))

# Simple safe photo path mapping
@app.route('/photos/<filename>')
def serve_photo(filename):
    return app.send_from_directory(UPLOAD_FOLDER, filename)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
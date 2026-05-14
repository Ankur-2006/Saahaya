import cv2
from deepface import DeepFace
import numpy as np
import time, requests
from twilio.rest import Client
import os

# -----------------------------
# Twilio configuration
# -----------------------------
account_sid = ''
auth_token = ''
twilio_client = Client(account_sid, auth_token)
twilio_from = ''
ALERT_NUMBERS = ''

# Flask endpoint running inside the woman's phone app
# Replace with your phone app’s endpoint or Firebase function
PHONE_ALERT_URL = "https://sahaaya-mobile-app.example.com/api/trigger_siren"

alert_cooldown = 60
last_alert_time = 0

def send_sms_alert():
    for num in ALERT_NUMBERS:
        twilio_client.messages.create(
            body="⚠️ Possible distress detected — please check immediately.",
            from_=twilio_from,
            to=num
        )

def place_siren_call():
    try:
        twilio_client.calls.create(
            twiml="<Response><Say voice='alice'>Emergency detected by Sahaaya. Please check immediately.</Say></Response>",
            to=ALERT_NUMBERS[0],
            from_=twilio_from
        )
    except Exception as e:
        print("Call error:", e)

def trigger_phone_siren():
    """Send a POST request to the phone app to play the siren locally."""
    try:
        requests.post(PHONE_ALERT_URL, json={"alert": True})
        print("📱 Triggered phone siren alert.")
    except Exception as e:
        print("Error triggering phone siren:", e)

def send_alerts_with_cooldown():
    global last_alert_time
    now = time.time()
    if now - last_alert_time > alert_cooldown:
        last_alert_time = now
        send_sms_alert()
        place_siren_call()
        trigger_phone_siren()
    else:
        print("⏱ Alert cooldown active — skipping duplicate alert.")

# -----------------------------
# Video analysis
# -----------------------------
cap = cv2.VideoCapture("walking.mp4")  # 0 for webcam
out = cv2.VideoWriter("output_with_alerts.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame,
                                  actions=['age', 'gender', 'emotion'],
                                  detector_backend='opencv',
                                  enforce_detection=False)
        faces = result if isinstance(result, list) else [result]
        men, women = 0, 0

        for f in faces:
            x, y, w, h = f['region']['x'], f['region']['y'], f['region']['w'], f['region']['h']
            gender = f['dominant_gender']
            emotion = f['dominant_emotion']

            if gender.lower() == 'man':
                men += 1
            elif gender.lower() == 'woman':
                women += 1
                distress = emotion.lower() in ("fear", "angry", "sad", "surprise")
                if distress and men >= 2:
                    cv2.putText(frame, "⚠️ Possible Distress Detected — Alert Sent", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    send_alerts_with_cooldown()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{gender}, {f['age']} yrs, {emotion}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Sahaaya AI Monitor", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

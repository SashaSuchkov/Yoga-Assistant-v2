# pip install opencv-python mediapipe numpy pygame

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import pygame

# --- НАЛАШТУВАННЯ АУДІО ---
pygame.mixer.init()
try:
    sound_pose_start = pygame.mixer.Sound("pose_ok.wav")     # Коли став у правильну позу
    sound_step_done = pygame.mixer.Sound("step_final.wav")   # Коли час кроку вичерпано
except:
    print("Попередження: Звукові файли не знайдено. Програма працюватиме без звуку.")
    sound_pose_start = sound_step_done = None

# --- КОНСТАНТИ ---
JSON_FILE = "warrior_ii.json"
BUFFER_ANGLE = 15.0
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
LANDMARK_MAP = {lm.name: lm.value for lm in mp_pose.PoseLandmark}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

class YogaApp:
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)["YogaPose"]
        self.mode = "PRACTICE"
        self.current_step_idx = 0
        self.step_start_time = None
        self.was_correct = False # Для відстеження моменту входу в позу
        
    def get_current_step(self):
        return self.data["steps"][self.current_step_idx]

    def validate_step(self, landmarks):
        step = self.get_current_step()
        all_correct = True
        feedback = []
        for angle_req in step.get("angles", []):
            try:
                p1 = [landmarks[LANDMARK_MAP[angle_req["joint1"]]].x, landmarks[LANDMARK_MAP[angle_req["joint1"]]].y]
                pb = [landmarks[LANDMARK_MAP[angle_req["base_joint"]]].x, landmarks[LANDMARK_MAP[angle_req["base_joint"]]].y]
                p2 = [landmarks[LANDMARK_MAP[angle_req["joint2"]]].x, landmarks[LANDMARK_MAP[angle_req["joint2"]]].y]
                curr = calculate_angle(p1, pb, p2)
                if not (angle_req["min_angle"] <= curr <= angle_req["max_angle"]):
                    all_correct = False
                    feedback.append(f"{angle_req['base_joint']}: {int(curr)}°")
            except: continue
        return all_correct, feedback

    def record_trainer_pose(self, landmarks):
        step = self.get_current_step()
        for angle_req in step.get("angles", []):
            p1 = [landmarks[LANDMARK_MAP[angle_req["joint1"]]].x, landmarks[LANDMARK_MAP[angle_req["joint1"]]].y]
            pb = [landmarks[LANDMARK_MAP[angle_req["base_joint"]]].x, landmarks[LANDMARK_MAP[angle_req["base_joint"]]].y]
            p2 = [landmarks[LANDMARK_MAP[angle_req["joint2"]]].x, landmarks[LANDMARK_MAP[angle_req["joint2"]]].y]
            measured = calculate_angle(p1, pb, p2)
            angle_req["min_angle"] = round(measured - BUFFER_ANGLE, 1)
            angle_req["max_angle"] = round(measured + BUFFER_ANGLE, 1)

# --- ЗАПУСК ---
app = YogaApp(JSON_FILE)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # 1. Фонова плашка інтерфейсу
    cv2.rectangle(frame, (0, 0), (w, 150), (20, 20, 20), -1)
    
    # 2. Вивід режиму та керування
    mode_color = (0, 255, 0) if app.mode == "PRACTICE" else (0, 165, 255)
    cv2.putText(frame, f"MODE: {app.mode}", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, mode_color, 2)
    cv2.putText(frame, "[M]-Switch Mode | [Q]-Exit", (20, 65), cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
    if app.mode == "TRAINER":
        cv2.putText(frame, "[R]-RECORD POSE FOR STEP", (20, 95), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

    # 3. Опис поточного кроку
    curr_step = app.get_current_step()
    cv2.putText(frame, f"STEP {curr_step['stepNumber']}/{len(app.data['steps'])}", (w//2 - 50, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(frame, curr_step['description'], (w//2 - 250, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        is_correct, errors = app.validate_step(results.pose_landmarks.landmark)

        if app.mode == "PRACTICE":
            if is_correct:
                # Звук А: щойно стали правильно
                if not app.was_correct:
                    if sound_pose_start: sound_pose_start.play()
                    app.was_correct = True
                
                if app.step_start_time is None: app.step_start_time = time.time()
                elapsed = time.time() - app.step_start_time
                rem = max(0, curr_step['duration'] - elapsed)
                
                # Таймер на екрані
                cv2.circle(frame, (w-80, 75), 40, (0, 255, 0), 2)
                cv2.putText(frame, str(int(rem)), (w-95, 85), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

                if rem <= 0:
                    # Звук Б: крок зараховано
                    if sound_step_done: sound_step_done.play()
                    if app.current_step_idx < len(app.data['steps']) - 1:
                        app.current_step_idx += 1
                        app.step_start_time = None
                        app.was_correct = False
            else:
                app.step_start_time = None
                app.was_correct = False
                # Підказки, що не так
                for i, err in enumerate(errors):
                    cv2.putText(frame, f"Fix: {err}", (w-200, 180 + i*30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)

    # Логіка кнопок
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('m'):
        app.mode = "TRAINER" if app.mode == "PRACTICE" else "PRACTICE"
        app.current_step_idx = 0
    if key == ord('r') and app.mode == "TRAINER" and results.pose_landmarks:
        app.record_trainer_pose(results.pose_landmarks.landmark)
        if app.current_step_idx < len(app.data['steps']) - 1:
            app.current_step_idx += 1
        else:
            with open(JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump({"YogaPose": app.data}, f, ensure_ascii=False, indent=2)
            app.mode = "PRACTICE"
            app.current_step_idx = 0

    cv2.imshow('Yoga AI: Professional Edition', frame)

cap.release()
cv2.destroyAllWindows()
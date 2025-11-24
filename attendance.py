import cv2
import face_recognition
import pickle
import os
import csv
from datetime import datetime
import numpy as np

ENC_PATH = "encodings.pkl"
ATTENDANCE_CSV = "attendance.csv"
TOLERANCE = 0.5

def load_encodings(enc_path=ENC_PATH):
    if not os.path.exists(enc_path):
        raise FileNotFoundError(f"Encodings file not found: {enc_path}. Run encode_faces.py first.")
    with open(enc_path, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

def ensure_csv(path=ATTENDANCE_CSV):
    if not os.path.exists(path):
        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

def already_marked(name, date_str, path=ATTENDANCE_CSV):
    if not os.path.exists(path):
        return False
    with open(path, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] == name and row["Date"] == date_str:
                return True
    return False

def mark_attendance(name, path=ATTENDANCE_CSV):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    if already_marked(name, date_str, path):
        print(f"[INFO] {name} already marked present for {date_str}")
        return False
    with open(path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])
    print(f"[INFO] Marked {name} at {date_str} {time_str}")
    return True

def main(camera_index=0):
    known_encodings, known_names = load_encodings()
    ensure_csv()

    video = cv2.VideoCapture(camera_index)
    print("[INFO] Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(distances) > 0:
                best_idx = np.argmin(distances)
                if distances[best_idx] <= TOLERANCE:
                    name = known_names[best_idx]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.rectangle(frame, (left, bottom-25), (right, bottom), (0,255,0), cv2.FILLED)
            cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

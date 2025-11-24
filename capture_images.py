import cv2
import os
import time

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_images(name, num_images=30, camera_index=0):
    dataset_dir = "dataset"
    person_dir = os.path.join(dataset_dir, name)
    create_dir(person_dir)

    cam = cv2.VideoCapture(camera_index)
    print(f"[INFO] Starting webcam. Press 'q' to quit early.")
    count = 0
    time.sleep(1.0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Can't read from camera")
            break

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//2 - 150, h//2 - 150), (w//2 + 150, h//2 + 150), (0,255,0), 2)
        cv2.putText(frame, f"Images captured: {count}/{num_images}  Press 'c' to capture", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Capture Images", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            crop = frame[h//2 - 150:h//2 + 150, w//2 - 150:w//2 + 150]
            cv2.imwrite(img_path, crop)
            print(f"[INFO] Saved {img_path}")
            count += 1
            if count >= num_images:
                print("[INFO] Reached target number of images.")
                break
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter person's name (no spaces, use underscores): ").strip()
    n = input("How many images (default 30)? ").strip()
    num_images = int(n) if n.isdigit() else 30
    capture_images(name, num_images=num_images)

import os
import face_recognition
import pickle
from imutils import paths

def encode_dataset(dataset_dir="dataset", encodings_path="encodings.pkl"):
    image_paths = list(paths.list_images(dataset_dir))
    known_encodings = []
    known_names = []

    print(f"[INFO] Found {len(image_paths)} images. Encoding...")
    for (i, image_path) in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] Processing {image_path}")
        image = face_recognition.load_image_file(image_path)
        boxes = face_recognition.face_locations(image, model="hog")
        encodings = face_recognition.face_encodings(image, boxes)
        name = image_path.split(os.path.sep)[-2]

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Saved encodings to {encodings_path}")

if __name__ == "__main__":
    encode_dataset()

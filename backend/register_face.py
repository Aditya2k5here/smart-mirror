from __future__ import annotations

import re

import cv2

from core.config import CONFIG


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned.strip("_") or "User"


def registration_guidance(frame, faces) -> tuple[str, tuple[int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    height, width = frame.shape[:2]

    if len(faces) == 0:
        return "Move into frame", (80, 170, 255)
    if len(faces) > 1:
        return "Only one face in frame", (80, 170, 255)
    if brightness < 70:
        return "Add more light", (80, 170, 255)
    if brightness > 205:
        return "Reduce glare", (80, 170, 255)

    x, y, w, h = faces[0]
    center_x = x + w / 2
    center_y = y + h / 2
    if abs(center_x - width / 2) > width * 0.18 or abs(center_y - height / 2) > height * 0.22:
        return "Center your face", (80, 170, 255)
    if w < width * 0.12:
        return "Move closer", (80, 170, 255)
    if w > width * 0.42:
        return "Move back slightly", (80, 170, 255)
    return "Ready - press SPACE", (130, 230, 180)


def register() -> None:
    print("=======================================")
    print(" Smart Mirror Face Registration")
    print("=======================================")
    name = input("Enter your name (example: Aditya): ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    save_dir = CONFIG.known_faces_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    image_path = save_dir / f"{safe_filename(name)}.jpg"

    cap = cv2.VideoCapture(CONFIG.camera_index)
    if not cap.isOpened():
        print(f"Could not open webcam index {CONFIG.camera_index}.")
        return

    print("\nStarting webcam. Look directly at the camera.")
    print("Press SPACE to capture your face, or ESC to cancel.")
    print(f"Saving into: {save_dir}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        guidance, color = registration_guidance(frame, faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.putText(
            frame,
            guidance,
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
        cv2.putText(
            frame,
            "SPACE: save   ESC: cancel",
            (18, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (220, 220, 220),
            2,
        )
        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1)

        if key == 32:
            guidance, _ = registration_guidance(frame, faces)
            if guidance != "Ready - press SPACE":
                print(f"Not saved yet: {guidance}")
                continue
            cv2.imwrite(str(image_path), frame)
            print(f"\nSuccess. Face registered for {name}.")
            print(f"Saved to {image_path}")
            break
        if key == 27:
            print("\nRegistration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register()

import cv2
import mediapipe as mp
import json

def extract_hand_landmarks(video_path, output_json='hand_landmarks.json'):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    landmarks_data = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        frame_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    frame_landmarks.append({
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z
                    })

        landmarks_data.append({
            'frame': frame_idx,
            'landmarks': frame_landmarks
        })
        frame_idx += 1

    cap.release()

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(landmarks_data, f, indent=2)

    print(f'Saved {frame_idx} frames of hand landmarks to {output_json}')
import cv2
import mediapipe as mp
import json
import numpy as np

def preprocess_frame(frame):
    # Giảm nhiễu
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    # Chuyển sang YUV để cân bằng sáng
    yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    # Cân bằng histogram (sáng)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    yuv_eq = cv2.merge((y_eq, u, v))
    enhanced = cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)
    
    return enhanced

def extract_hand_landmarks(video_path, output_json='hand_landmarks.json'):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    landmarks_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Tiền xử lý khung hình
        frame = preprocess_frame(frame)

        # Đổi BGR sang RGB
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

    with open(output_json, 'w') as f:
        json.dump(landmarks_data, f, indent=2)

    print(f'Saved {frame_idx} frames of hand landmarks to {output_json}')
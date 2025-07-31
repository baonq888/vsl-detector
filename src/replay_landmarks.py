import cv2
import json

def replay_landmarks_on_video(video_path, json_file, pause=1):
    # Load landmark data
    with open(json_file, 'r') as f:
        data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        return

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(data):
            break

        frame_data = data[frame_idx]
        landmarks = frame_data.get('landmarks', [])

        h, w, _ = frame.shape  # height and width for coordinate scaling

        # Draw landmarks
        for i, lm in enumerate(landmarks):
            cx, cy = int(lm['x'] * w), int(lm['y'] * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)  # green dot
            cv2.putText(frame, str(i), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Show the frame
        cv2.imshow("Replay with Landmarks", frame)
        if cv2.waitKey(int(1000 / cap.get(cv2.CAP_PROP_FPS))) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if __name__ == "__main__":
        video_path = "data/video_sign_language.mp4"
        json_file = "data/output.json"
        replay_landmarks_on_video(video_path, json_file)
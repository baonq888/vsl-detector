from hand_landmark_extraction import extract_hand_landmarks
from replay_landmarks import replay_landmarks_on_video

extract_hand_landmarks("data/video_sign_language.mp4", "data/output.json")

replay_landmarks_on_video("data/video_sign_language.mp4","data/output.json")

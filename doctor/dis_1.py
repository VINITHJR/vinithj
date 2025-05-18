import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time


class HealthConditionDetector:
    def __init__(self, seizure_motion_threshold=0.03, seizure_history_length=10, seizure_duration=2.0,
                 chest_history_length=15, chest_proximity_threshold=0.35, head_movement_threshold=0.02,
                 eye_squint_threshold=0.01, brow_tension_threshold=0.02, mouth_width_threshold=0.2,
                 chest_pain_confidence_threshold=0.33, chest_min_detection_seconds=0.25,
                 fall_threshold=0.03, fall_history_length=5):
        # Initialize Mediapipe
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils

        # Seizure Detection Parameters
        self.seizure_motion_threshold = seizure_motion_threshold
        self.seizure_history_length = seizure_history_length
        self.seizure_duration = seizure_duration
        self.seizure_detected = False
        self.wrist_history_left = deque(maxlen=seizure_history_length)
        self.wrist_history_right = deque(maxlen=seizure_history_length)
        self.nose_history = deque(maxlen=seizure_history_length)
        self.seizure_start_time = None
        self.fps = 30  # Default, updated from video

        # Chest Pain Detection Parameters
        self.chest_history_length = chest_history_length
        self.chest_proximity_threshold = chest_proximity_threshold
        self.head_movement_threshold = head_movement_threshold
        self.eye_squint_threshold = eye_squint_threshold
        self.brow_tension_threshold = brow_tension_threshold
        self.mouth_width_threshold = mouth_width_threshold
        self.chest_pain_confidence_threshold = chest_pain_confidence_threshold
        self.chest_min_detection_seconds = chest_min_detection_seconds
        self.nose_x_history = deque(maxlen=chest_history_length)
        self.wrist_near_chest_history = deque(maxlen=5)
        self.consecutive_chest_frames = 0

        # Fall Detection Parameters
        self.fall_threshold = fall_threshold
        self.fall_history_length = fall_history_length
        self.fall_detected = False
        self.y_position_history = deque(maxlen=fall_history_length)

    def _calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def _calculate_midpoint(self, point1, point2, shoulder_weight=0.7):
        class Point:
            pass
        midpoint = Point()
        midpoint.x = (point1.x * shoulder_weight + point2.x * (1 - shoulder_weight))
        midpoint.y = (point1.y * shoulder_weight + point2.y * (1 - shoulder_weight))
        midpoint.z = (point1.z * shoulder_weight + point2.z * (1 - shoulder_weight)) if hasattr(point1, 'z') and hasattr(point2, 'z') else 0
        return midpoint

    def _is_whole_body_visible(self, landmarks):
        required_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        return all(landmarks[lm].visibility > 0.5 for lm in required_landmarks)

    def _is_upper_body_visible(self, landmarks):
        required_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]
        return all(landmarks[lm].visibility > 0.5 for lm in required_landmarks)

    # Seizure Detection Methods
    def _calculate_motion_score(self, history):
        if len(history) < self.seizure_history_length:
            return 0.0
        x_coords = [pos[0] for pos in history]
        y_coords = [pos[1] for pos in history]
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        motion_score = max(x_std, y_std) / self.seizure_motion_threshold
        return min(1.0, motion_score)

    def _is_seizure(self, wrist_motion_left, wrist_motion_right, head_motion):
        return (wrist_motion_left > 0.7 or wrist_motion_right > 0.7 or head_motion > 0.7)

    # Chest Pain Detection Methods
    def _is_wrist_on_chest(self, landmarks):
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_mid = self._calculate_midpoint(left_shoulder, right_shoulder)
        hip_mid = self._calculate_midpoint(left_hip, right_hip)
        chest_center = self._calculate_midpoint(shoulder_mid, hip_mid, shoulder_weight=0.7)

        left_dist = self._calculate_distance(left_wrist, chest_center)
        right_dist = self._calculate_distance(right_wrist, chest_center)
        min_dist = min(left_dist, right_dist)

        base_score = 1.0 - min(1.0, min_dist / self.chest_proximity_threshold)

        penalty = 1.0
        if (left_dist <= self.chest_proximity_threshold and right_dist > self.chest_proximity_threshold) or \
           (right_dist <= self.chest_proximity_threshold and left_dist > self.chest_proximity_threshold):
            penalty *= 0.5
        shoulder_y = shoulder_mid.y
        chest_y = chest_center.y
        wrist_y = left_wrist.y if left_dist < right_dist else right_wrist.y
        if wrist_y < chest_y and (chest_y - wrist_y) / (chest_y - shoulder_y) > 0.5:
            penalty *= 0.7

        final_score = base_score * penalty
        wrist_near_chest = min_dist <= self.chest_proximity_threshold
        self.wrist_near_chest_history.append(wrist_near_chest)
        return final_score, min_dist

    def _calculate_head_movement(self):
        if len(self.nose_x_history) < 5:
            return 0.1
        x_coords = list(self.nose_x_history)[-5:]
        x_std = np.std(x_coords)
        return min(1.0, x_std / self.head_movement_threshold)

    def _calculate_facial_discomfort(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(frame_rgb)
        if not face_results.multi_face_landmarks:
            return 0.0

        for face_landmarks in face_results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[263]
            interocular_distance = self._calculate_distance(left_eye_outer, right_eye_outer)

            if interocular_distance == 0:
                return 0.0

            left_eye_dist = self._calculate_distance(landmarks[159], landmarks[145])
            right_eye_dist = self._calculate_distance(landmarks[386], landmarks[374])
            avg_eye_dist = (left_eye_dist + right_eye_dist) / 2
            normalized_eye = avg_eye_dist / interocular_distance
            eye_squint_score = 1.0 - min(1.0, normalized_eye / self.eye_squint_threshold)

            left_brow_dist = abs(landmarks[70].y - landmarks[33].y)
            right_brow_dist = abs(landmarks[300].y - landmarks[263].y)
            avg_brow_dist = (left_brow_dist + right_brow_dist) / 2
            normalized_brow = avg_brow_dist / interocular_distance
            brow_tension_score = 1.0 - min(1.0, normalized_brow / self.brow_tension_threshold)

            mouth_width = abs(landmarks[61].x - landmarks[291].x)
            normalized_mouth = mouth_width / interocular_distance
            mouth_tension_score = min(1.0, normalized_mouth / self.mouth_width_threshold)

            discomfort_score = (eye_squint_score * 0.3 + brow_tension_score * 0.5 + mouth_tension_score * 0.2)
            return discomfort_score
        return 0.0

    def _calculate_chest_pain_confidence(self, wrist_score, head_score, facial_score):
        confidence = (wrist_score * 0.5 + head_score * 0.1 + facial_score * 0.4)
        return confidence

    def _draw_chest_debug_visualization(self, frame, pose_results, wrist_distance, confidence, facial_score, required_chest_frames):
        debug_frame = frame.copy()
        self.mp_drawing.draw_landmarks(debug_frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        indicator_color = (0, 255, 0) if confidence > self.chest_pain_confidence_threshold else (0, 0, 255)

        cv2.putText(debug_frame, f"Wrist-Chest Distance: {wrist_distance:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Chest Pain Confidence: {confidence:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)
        cv2.putText(debug_frame, f"Facial Discomfort Score: {facial_score:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if confidence > self.chest_pain_confidence_threshold:
            cv2.putText(debug_frame, f"Possible Chest Pain: {self.consecutive_chest_frames}/{required_chest_frames} frames",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return debug_frame

    # Fall Detection Method
    def _is_falling(self, y_position):
        self.y_position_history.append(y_position)
        if len(self.y_position_history) < self.fall_history_length:
            return False
        y_diffs = [self.y_position_history[i] - self.y_position_history[i-1]
                   for i in range(1, len(self.y_position_history))]
        avg_movement = np.mean([diff for diff in y_diffs if diff > 0])  # Only consider downward movements
        return avg_movement > self.fall_threshold if not np.isnan(avg_movement) else False

    # Main Processing Method
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        required_chest_frames = max(1, int(self.fps * self.chest_min_detection_seconds))
        chest_pain_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(frame_rgb)
            status_text = "No Condition Detected"
            status_color = (0, 255, 0)
            debug_frame = frame

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                upper_body_visible = self._is_upper_body_visible(landmarks)

                # Seizure Detection (requires upper body: nose, shoulders, wrists)
                if upper_body_visible:
                    left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                    self.wrist_history_left.append((left_wrist.x, left_wrist.y))
                    self.wrist_history_right.append((right_wrist.x, right_wrist.y))
                    self.nose_history.append((nose.x, nose.y))

                    wrist_motion_left = self._calculate_motion_score(self.wrist_history_left)
                    wrist_motion_right = self._calculate_motion_score(self.wrist_history_right)
                    head_motion = self._calculate_motion_score(self.nose_history)

                    if self._is_seizure(wrist_motion_left, wrist_motion_right, head_motion):
                        if self.seizure_start_time is None:
                            self.seizure_start_time = time.time()
                        elif time.time() - self.seizure_start_time >= self.seizure_duration:
                            self.seizure_detected = True
                            status_text = "Seizure Detected"
                            status_color = (0, 0, 255)
                    else:
                        self.seizure_start_time = None
                else:
                    self.seizure_start_time = None

                # Fall Detection (requires whole body)
                if self._is_whole_body_visible(landmarks):
                    y_position = landmarks[self.mp_pose.PoseLandmark.NOSE].y
                    if self._is_falling(y_position):
                        self.fall_detected = True
                        status_text = "Fall Detected" if status_text == "No Condition Detected" else status_text + " & Fall"
                        status_color = (0, 0, 255)
                else:
                    self.y_position_history.clear()

                # Chest Pain Detection (requires upper body)
                if upper_body_visible:
                    self.nose_x_history.append(landmarks[self.mp_pose.PoseLandmark.NOSE].x)
                    wrist_score, wrist_distance = self._is_wrist_on_chest(landmarks)
                    facial_score = self._calculate_facial_discomfort(frame)
                    wrist_was_near_chest = any(self.wrist_near_chest_history)
                    confidence = 0.0

                    if facial_score == 0.0 or not wrist_was_near_chest:
                        self.consecutive_chest_frames = 0
                        debug_frame = frame
                    else:
                        head_score = self._calculate_head_movement()
                        confidence = self._calculate_chest_pain_confidence(wrist_score, head_score, facial_score)
                        if confidence > self.chest_pain_confidence_threshold:
                            self.consecutive_chest_frames += 1
                            if self.consecutive_chest_frames >= required_chest_frames:
                                chest_pain_detected = True
                                status_text = "Chest Pain Detected" if status_text == "No Condition Detected" else status_text + " & Chest Pain"
                                status_color = (0, 0, 255)
                        else:
                            self.consecutive_chest_frames = 0
                        debug_frame = self._draw_chest_debug_visualization(frame, pose_results, wrist_distance, confidence, facial_score, required_chest_frames)
                else:
                    self.consecutive_chest_frames = 0
                    status_text = "Upper Body Not Visible" if status_text == "No Condition Detected" else status_text
                    status_color = (255, 0, 0)
                    confidence = 0.0
                    debug_frame = frame

            if chest_pain_detected:
                cv2.putText(debug_frame, "CHEST PAIN DETECTED", (int(debug_frame.shape[1] / 2) - 150, int(debug_frame.shape[0] / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.putText(debug_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow("Health Condition Detector", debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        
        cv2.destroyAllWindows()
        return self.seizure_detected or self.fall_detected or chest_pain_detected

if __name__ == "__main__":
    video_path = r"C:\Users\VINITH J R\OneDrive\Desktop\doctor\falling.mp4"  # Replace with your actual video path
    detector = HealthConditionDetector(fall_threshold=0.03, chest_min_detection_seconds=0.25)
    result = detector.process_video(video_path)
    print(result)
    
from flask import Flask, request, jsonify
import threading
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

detector = HealthConditionDetector()

@app.route('/run_detection', methods=['POST'])
def run_detection():
    data = request.json
    video_path = data.get('video_path')
    if not video_path:
        return jsonify({'error': 'Missing video_path'}), 400

    def process():
        result = detector.process_video(video_path)
        detection_results[video_path] = result

    detection_results = {}
    thread = threading.Thread(target=process)
    thread.start()
    thread.join()

    result = detection_results.get(video_path, False)
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)

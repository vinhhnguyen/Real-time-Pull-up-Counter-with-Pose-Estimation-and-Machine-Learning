import csv, cv2
import mediapipe as mp

# Set up Mediapipe pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Set up video capture
cap = cv2.VideoCapture("C:/Users/vinhh/Project Code/Rep Counter/david goggins pull up videos.mp4")

# List of landmarks
landmarks = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_PINKY,
    mp_pose.PoseLandmark.RIGHT_PINKY,
    mp_pose.PoseLandmark.LEFT_INDEX,
    mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.LEFT_THUMB,
    mp_pose.PoseLandmark.RIGHT_THUMB,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,]


# Set up CSV output file
with open("landmark coordinates.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["stage_position"]
    for landmark in landmarks:
        header += [f"{landmark.name}_x", f"{landmark.name}_y", f"{landmark.name}_z", f"{landmark.name}_visibility"]
    writer.writerow(header)

    # Loop through video frames and extract pose data
    while True:
        # Read frame from video
        ret, image = cap.read()
        if not ret:
            break

        # Process frame with Mediapipe pose estimation model
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Extract stage position and landmark data
        if results.pose_landmarks is not None:
            # Get position of relevant body landmarks
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Determine whether subject is in up or down position
            if (
                left_shoulder.y > left_elbow.y and left_elbow.y > left_wrist.y
            ) or (
                right_shoulder.y > right_elbow.y and right_elbow.y > right_wrist.y
            ):
                stage_position = "down"
            else:
                stage_position = "up"

            # Extract landmark data
            landmark_coord = [stage_position]
            for landmark in landmarks:
                landmark_coordinates = results.pose_landmarks.landmark[landmark]
                landmark_coord += [landmark_coordinates.x, landmark_coordinates.y, landmark_coordinates.z, landmark_coordinates.visibility]

            # Write data to CSV file
            writer.writerow(landmark_coord)

        # Draw pose landmarks on frame
        if results.pose_landmarks is not None:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Render stage value
        cv2.rectangle(image, (80, 10), (200, 70), (245, 117, 16), -1)
        cv2.putText(image, 'STAGE', (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, stage_position, (100, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

        # Display frame with pose landmarks
        cv2.imshow("Pull-Up Video", image)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) == ord("q"):
            break

# Release video capture and pose estimation resources
cap.release()
pose.close()
cv2.destroyAllWindows()

from flask import Flask, render_template, request, redirect, url_for, session, Response
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import os
from datetime import datetime
from PIL import Image

app = Flask(__name__)
app.secret_key = "this_have_been_developing_by_jinx"


# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
# SQLite Database Setup
# Environment variables for SQLite database path
DATABASE_PATH = os.getenv("DATABASE_PATH", "angle_data.db")


# Initialize the database
def init_db():
    db_exists = os.path.exists(DATABASE_PATH)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    if not db_exists:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS angle_data (
                joint_name TEXT,
                min_angle REAL,
                max_angle REAL,
                timestamp TEXT
            )
        """)
        conn.commit()
    return conn, cursor

def update_angle_data(cursor, joint_name, min_angle, max_angle):
    """Insert or update angle data in the SQLite database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('SELECT id FROM angle_data WHERE joint_name = ?', (joint_name,))
    record = cursor.fetchone()
    if record:
        cursor.execute('''UPDATE angle_data SET min_angle = ?, max_angle = ?, timestamp = ? WHERE id = ?''',
                       (min_angle, max_angle, timestamp, record[0]))
    else:
        cursor.execute('''INSERT INTO angle_data (joint_name, min_angle, max_angle, timestamp)
                          VALUES (?, ?, ?, ?)''', (joint_name, min_angle, max_angle, timestamp))
        
def calculate_angle_points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def calculate_flex_extension(shoulder, wrist):
    vertical_line = [shoulder[0], shoulder[1] + 100]
    limb_angle = calculate_angle_points(wrist, shoulder, vertical_line)

    # Extension occurs when the angle is less than 180°
    if limb_angle < 180:
        return 180 - limb_angle
    return 0

def calculate_angle_ref(reference, target):
    reference = np.array(reference)
    target = np.array(target)
    dot_product = np.dot(reference, target)
    magnitude_reference = np.linalg.norm(reference)
    magnitude_target = np.linalg.norm(target)
    cosine_angle = dot_product / (magnitude_reference * magnitude_target + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_frame(results, cursor):
    """Process the frame to calculate and update neck angles."""
    landmarks = results.pose_landmarks.landmark
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    nose = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
    neck = [(shoulder_left[0] + shoulder_right[0]) / 2, (shoulder_left[1] + shoulder_right[1]) / 2]

    neck_angles = {
        'neck_lateral_left': 90 - calculate_angle_points(nose, neck, shoulder_left),
        'neck_lateral_right': 90 - calculate_angle_points(nose, neck, shoulder_right)
    }

    for name, angle in neck_angles.items():
        max_angle = max(cursor.execute('SELECT IFNULL(MAX(max_angle), 0) FROM angle_data WHERE joint_name = ?', (name,)).fetchone()[0], angle)
        min_angle = 0
        update_angle_data(cursor, name, min_angle, max_angle)
    
    return neck_angles

def display_angles(image, neck_angles, max_min_angles):
    """Overlay neck angles on the frame."""
    for name, angle in neck_angles.items():
        cv2.putText(image, f"{name}: {int(angle)}", (10, 30 + 30 * list(neck_angles.keys()).index(name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display max/min angles
    y_position = 300
    for name, values in max_min_angles.items():
        cv2.putText(image, f"{name} Max: {int(values['max'])}", (10, y_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_position += 20

def track_elbow_left(conn,cursor):
    max_angle = 0  # To track the maximum angle
    min_angle = 180
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Get pose landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of shoulder, elbow, and wrist (left side)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate the original angle
            elbow = 180 - calculate_angle_points(shoulder, elbow, wrist)
            
            # Angle should be between 0 and 160 degrees
            # if elbow > 140:
            #     elbow = 140

            # Update the maximum angle
            max_angle = max(max_angle, elbow)
            min_angle = min(min_angle, elbow)
            update_angle_data(cursor,"left_elbow", min_angle, max_angle)
            conn.commit()
            # Display the angles on the frame
            cv2.putText(frame, f'Left Elbow Flexion: {int(elbow)}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Min Angle: {int(min_angle)}', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Max: {int(max_angle)}', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        #cv2.imshow('Left Elbow Angle Tracker', frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Break the loop if 'q' is pressed
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break

    # Release resources
    cap.release()
    cv2.destroyAllWindows() 

def track_elbow_right(conn,cursor):
    max_angle = 0  # To track the maximum angle
    min_angle = 180
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Get pose landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of shoulder, elbow, and wrist (right side)
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate the angle
            angle = 180 - calculate_angle_points(shoulder, elbow, wrist)

            # Update the maximum and minimum angle
            max_angle = max(max_angle, angle)
            min_angle = min(min_angle, angle)
            update_angle_data(cursor,"right_elbow", min_angle, max_angle)
            conn.commit()
            # Display the angles on the frame
            cv2.putText(frame, f'Right Elbow Flexion: {int(angle)}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Min Angle: {int(min_angle)}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Max Angle: {int(max_angle)}', (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        #cv2.imshow('Elbow Angle Tracker', frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def track_extension_left(conn,cursor):

    max_extension = 0  # Maximum forward angle
    min_extension = float("inf")  # Minimum forward angle (initialize with infinity)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture the video")
            break

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract frame dimensions for scaling landmarks
            frame_height, frame_width, _ = frame.shape

            # Get coordinates for shoulder and wrist (left side as example)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height]

            # Calculate extension
            extension = 180 - calculate_flex_extension(shoulder, wrist)

            # Update maximum and minimum values
            max_extension = max(max_extension, extension)
            min_extension = min(min_extension, extension)

            update_angle_data(cursor,"left_extension_limb", min_extension, max_extension)
            conn.commit()
            # Display angles and max extension
            cv2.putText(frame, f"Extension: {extension:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Max Extension: {max_extension:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Draw key lines for visualization
            cv2.line(frame, tuple(map(int, shoulder)), (int(shoulder[0]), int(shoulder[1] + 100)), (0, 255, 0), 2)  # Vertical body line
            cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, wrist)), (255, 0, 0), 2)  # Arm line

            # Draw key points
            cv2.circle(frame, tuple(map(int, shoulder)), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(map(int, wrist)), 5, (255, 0, 0), -1)

        # Show the frame
        #cv2.imshow('Upper Limb Extension Tracking', frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # # Quit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #cv2.putText(frame, "Tracking Elbow Left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def track_extension_right(conn,cursor):

    max_extension = 0  # Maximum forward angle
    min_extension = float("inf")  # Minimum forward angle (initialize with infinity)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract frame dimensions for scaling landmarks
            frame_height, frame_width, _ = frame.shape

            # Get coordinates for shoulder and wrist (right side as example)
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height]

            # Calculate extension
            extension = 180 - calculate_flex_extension(shoulder, wrist)

            # Update maximum and minimum values
            max_extension = max(max_extension, extension)
            min_extension = min(min_extension, extension)

            update_angle_data(cursor,"right_extension_limb", min_extension, max_extension)
            conn.commit()
            # Display angles and max extension
            cv2.putText(frame, f"Extension: {extension:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Max Extension: {max_extension:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Draw key lines for visualization
            cv2.line(frame, tuple(map(int, shoulder)), (int(shoulder[0]), int(shoulder[1] + 100)), (0, 255, 0), 2)  # Vertical body line
            cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, wrist)), (255, 0, 0), 2)  # Arm line

            # Draw key points
            cv2.circle(frame, tuple(map(int, shoulder)), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(map(int, wrist)), 5, (255, 0, 0), -1)

        # Show the frame
        #cv2.imshow('Upper Limb Extension Tracking', frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Quit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def track_flexion_left(conn,cursor):

    max_flexion = 0  # Maximum forward angle
    min_flexion = float("inf")  # Minimum forward angle (initialize with infinity)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            break

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract frame dimensions for scaling landmarks
            frame_height, frame_width, _ = frame.shape

            # Get coordinates for shoulder and wrist (right side as example)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height]

            # Calculate flexion
            flexion = 180 - calculate_flex_extension(shoulder, wrist)

            # Update maximum and minimum values
            max_flexion = max(max_flexion, flexion)
            min_flexion = min(min_flexion, flexion)

            update_angle_data(cursor,"left_flexion_limb", min_flexion, max_flexion)
            conn.commit()
            # Display angles and max flexion
            cv2.putText(frame, f"Flexion: {flexion:.2f}°", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Max Flexion: {max_flexion:.2f}°", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Draw key lines for visualization
            cv2.line(frame, tuple(map(int, shoulder)), (int(shoulder[0]), int(shoulder[1] + 100)), (0, 255, 0), 2)  # Vertical body line
            cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, wrist)), (255, 0, 0), 2)  # Arm line

            # Draw key points
            cv2.circle(frame, tuple(map(int, shoulder)), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(map(int, wrist)), 5, (255, 0, 0), -1)

        # Show the frame
        #cv2.imshow('Upper Limb Flexion Tracking', frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Quit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def track_flexion_right(conn,cursor):

    max_flexion = 0  # Maximum forward angle
    min_flexion = float("inf")  # Minimum forward angle (initialize with infinity)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video.")
            break

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract frame dimensions for scaling landmarks
            frame_height, frame_width, _ = frame.shape

            # Get coordinates for shoulder and wrist (right side as example)
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height]

            # Calculate flexion
            flexion = 180 - calculate_flex_extension(shoulder, wrist)

            # Update maximum and minimum values
            max_flexion = max(max_flexion, flexion)
            min_flexion = min(min_flexion, flexion)

            update_angle_data(cursor,"right_flexion_limb", min_flexion, max_flexion)
            conn.commit()
            # Display angles and max flexion
            cv2.putText(frame, f"Flexion: {flexion:.2f}°", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Max Flexion: {max_flexion:.2f}°", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Draw key lines for visualization
            cv2.line(frame, tuple(map(int, shoulder)), (int(shoulder[0]), int(shoulder[1] + 100)), (0, 255, 0), 2)  # Vertical body line
            cv2.line(frame, tuple(map(int, shoulder)), tuple(map(int, wrist)), (255, 0, 0), 2)  # Arm line

            # Draw key points
            cv2.circle(frame, tuple(map(int, shoulder)), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(map(int, wrist)), 5, (255, 0, 0), -1)

        # Show the frame
        #cv2.imshow('Upper Limb Flexion Tracking', frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Quit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def track_flexion_reflexion(conn,cursor):

    # Tracking variables
    min_flexion = 0  # Minimum angle for flexion
    max_reflexion = 0  # Maximum angle for reflexion

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to access the camera.")
            break

        # Flip the frame for natural side pose view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(rgb_frame)

        if results_pose.pose_landmarks:
            # Get frame dimensions
            frame_height, frame_width, _ = frame.shape

            # Extract key landmarks
            landmarks = results_pose.pose_landmarks.landmark
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * frame_width,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y * frame_height]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]

            # Compute the ear-to-shoulder line
            ear_to_shoulder_line = [right_shoulder[0] - right_ear[0], right_shoulder[1] - right_ear[1]]

            # Vertical reference vector
            vertical_reference = [0, -1]  # Vertical line in screen coordinates

            # Calculate angle with respect to the vertical line
            angle = calculate_angle_ref(vertical_reference, ear_to_shoulder_line) - 150

            # Determine flexion and reflexion
            min_flexion = min(min_flexion, angle)
            max_reflexion = max(max_reflexion, angle)

            # Save the angles to the database
            update_angle_data(cursor, "neck_flex_reflex", abs(min_flexion), max_reflexion)
            conn.commit()

            # Display information on the frame
            cv2.putText(frame, f"Flexion (Min): {abs(min_flexion):.2f}°", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Reflexion (Max): {max_reflexion:.2f}°", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Current Angle: {abs(angle):.2f}°", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Draw the ear-to-shoulder line and vertical reference
            cv2.line(frame, tuple(map(int, right_ear)), tuple(map(int, right_shoulder)), (255, 0, 0), 2)  # Ear-to-shoulder line
            cv2.line(frame, tuple(map(int, right_ear)), (int(right_ear[0]), int(right_ear[1] - 50)), (0, 255, 0), 2)  # Vertical reference

            # Draw key landmarks
            cv2.circle(frame, tuple(map(int, right_ear)), 5, (0, 255, 0), -1)  # Right ear
            cv2.circle(frame, tuple(map(int, right_shoulder)), 5, (255, 0, 0), -1)  # Right shoulder

        # Display the frame
        #cv2.imshow("Flexion and Reflexion Tracker", frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    conn.close()
    cv2.destroyAllWindows()

def track_neck_lateral(conn,cursor):

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process frame and calculate angles
            if results.pose_landmarks:
                neck_angles = process_frame(results, cursor)
                conn.commit()
                
                # Get max and min angles from the database for display
                max_min_angles = {name: {'max': cursor.execute('SELECT max_angle FROM angle_data WHERE joint_name = ?', (name,)).fetchone()[0]}
                                  for name in neck_angles.keys()}
                
                # Display angles
                display_angles(image, neck_angles, max_min_angles)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Show frame
            cv2.imshow('Neck Angle Tracking', image)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

def track_neck_rotation(conn,cursor):

    max_left_deviation = 0  # Track max deviation for the left
    max_right_deviation = 0  # Track max deviation for the right

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video.")
            break

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract frame dimensions for scaling landmarks
            frame_height, frame_width, _ = frame.shape

            # Get coordinates for shoulders and nose
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x * frame_width,
                    landmarks[mp_pose.PoseLandmark.NOSE].y * frame_height]

            # Calculate midpoint of shoulders
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2]

            # Calculate angle deviations for left and right separately
            left_deviation = calculate_angle_points(nose, mid_shoulder, left_shoulder) - 90
            right_deviation = calculate_angle_points(nose, mid_shoulder, right_shoulder) - 90

            # Track max deviations independently
            max_left_deviation = max(max_left_deviation, left_deviation)
            max_right_deviation = max(max_right_deviation, right_deviation)

            # Draw the shoulder line
            cv2.line(frame, tuple(map(int, left_shoulder)), tuple(map(int, right_shoulder)), (0, 255, 0), 2)

            # Draw the perpendicular line to the nose
            cv2.line(frame, tuple(map(int, mid_shoulder)), tuple(map(int, nose)), (0, 255, 255), 2)

            # Draw points
            cv2.circle(frame, tuple(map(int, left_shoulder)), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(map(int, right_shoulder)), 5, (0, 0, 255), -1)
            cv2.circle(frame, tuple(map(int, mid_shoulder)), 5, (0, 255, 0), -1)
            cv2.circle(frame, tuple(map(int, nose)), 5, (0, 255, 255), -1)

            # Display current deviations and max deviations
            cv2.putText(frame, f"Left Rotation: {right_deviation:.2f} deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Right Rotation: {left_deviation:.2f} deg", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Max Left: {max_right_deviation:.2f} deg", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"Max Right: {max_left_deviation:.2f} deg", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show the frame
        #cv2.imshow('Neck Rotation Tracking', frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        update_angle_data(cursor,"neck_rotation", max_left_deviation, max_right_deviation)
        conn.commit()
        
        # Quit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Mapping function names to functions
TRACK_FUNCTIONS = {
    "track_elbow_right": (track_elbow_right, {"conn": None, "cursor": None}),
    "track_elbow_left": (track_elbow_left, {"conn": None, "cursor": None}),
    "track_extension_left": (track_extension_left, {"conn": None, "cursor": None}),
    "track_extension_right": (track_extension_right, {"conn": None, "cursor": None}),
    "track_flexion_left": (track_flexion_left, {"conn": None, "cursor": None}),
    "track_flexion_right": (track_flexion_right, {"conn": None, "cursor": None}),
    "track_flexion_reflexion": (track_flexion_reflexion, {"conn": None, "cursor": None}),
    "track_neck_lateral": (track_neck_lateral, {"conn": None, "cursor": None}),
    "track_neck_rotation": (track_neck_rotation, {"conn": None, "cursor": None}),
}


@app.route("/")
def main_page():
    # Options to display on the main page
    options = list(TRACK_FUNCTIONS.keys())
    return render_template("main.html", options=options)

@app.route("/track/<function_name>")
def track(function_name):
    if function_name in TRACK_FUNCTIONS:
        return render_template("track.html", function_name=function_name)
    else:
        return "Invalid function name", 404

@app.route("/video_feed/<function_name>")
def video_feed(function_name):
    func_info = TRACK_FUNCTIONS.get(function_name)
    if func_info:
        func, default_args = func_info
        conn, cursor = init_db()  # Initialize the database connection
        updated_args = {**default_args, "conn": conn, "cursor": cursor}

        return Response(func(**updated_args), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid function name", 404

def fetch_all_tracking_data():
    conn,cursor = init_db()
    cursor.execute("SELECT joint_name, min_angle, max_angle, timestamp FROM angle_data")
    results = cursor.fetchall()  # Fetch all results
    conn.close()  # Close the connection
    return results

@app.route('/results')
def results():
    tracking_data = fetch_all_tracking_data()  # Get data from the database
    return render_template("results.html", tracking_data=tracking_data)

@app.route('/test_video_feed/<function_name>')
def test_video_feed(function_name):
    func_info = TRACK_FUNCTIONS.get(function_name)
    if func_info:
        func, default_args = func_info
        conn, cursor = init_db()  # Initialize the database connection
        updated_args = {**default_args, "conn": conn, "cursor": cursor}
    def generate_frames():
        for frame in func(**updated_args):
                # Encode the frame in JPEG format
                _, buffer = cv2.imencode('.png', frame)
                frame = buffer.tobytes()
                cv2.imshow('',frame)
                #yield (b'--frame\r\n'
                #      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        conn.close()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_csv')
def download_csv():
    conn, cursor = init_db()
    cursor.execute("SELECT joint_name, min_angle, max_angle, timestamp FROM angle_data")
    data = cursor.fetchall()
    conn.close()
    
    # Define header for CSV
    csv_output = [["Joint Name", "Minimum Angle", "Maximum Angle", "Last Updated"]]
    for row in data:
        formatted_row = [
            row[0],  # Joint Name
            row[1],  # Minimum Angle
            row[2],  # Maximum Angle
            f"'{row[3]}"  # Force Timestamp to Text
        ]
        csv_output.append(formatted_row)
    
    # Create a CSV response
    def generate_csv():
        for row in csv_output:
            yield ','.join([str(field) for field in row]) + '\n'

    # Set response headers for downloading
    response = Response(generate_csv(), mimetype='text/csv')
    response.headers.set("Content-Disposition", "attachment", filename="tracking_data.csv")
    return response

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

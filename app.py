from flask import Flask, render_template, request, redirect, url_for, session, Response, send_file, jsonify
import cv2
import csv
import mediapipe as mp
import numpy as np
import sqlite3, time
from datetime import datetime,time
from PIL import Image
from io import BytesIO
import warnings
import json
import base64
import os

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

app = Flask(__name__)
app.secret_key = "this_have_been_developing_by_jinx"


# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

CSV_EXPORT_DIR = "static\csv_exports"

# Ensure the directory exists
os.makedirs(CSV_EXPORT_DIR, exist_ok=True)

# JSON file to store config variables
CONFIG_FILE = "config.json"

# Function to load configuration from JSON file
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            return json.load(file)
    else:
        return {"id": 0, "toggle": False}

# Function to save configuration to JSON file
def save_config(config):
    with open(CONFIG_FILE, "w") as file:
        json.dump(config, file, indent=4)

# Load config at startup
config = load_config()
id = config.get("id", 0)  # Get stored ID, default to 0 if not found
#toggle = config.get("toggle", False)  # Get stored toggle, default to False if not found

def update_config(new_id):#, new_toggle):
    config["id"] = new_id
    #config["toggle"] = new_toggle
    save_config(config)

#Global Toggle Option
toggle = False

# SQLite Database Setup
def init_db():
    conn = sqlite3.connect("angle_data.db")
    cursor = conn.cursor()

    # Create a table to store limb data if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS angle_data (
        id INTEGER,
        joint_name TEXT NOT NULL,
        min_angle REAL,
        max_angle REAL,
        timestamp TEXT
    )
    """)
    conn.commit()
    return conn,cursor

def update_angle_data(cursor, joint_name, min_angle, max_angle):
    global id
    update_config(id)
    """Insert or update angle data in the SQLite database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('SELECT id FROM angle_data WHERE id = ? AND joint_name = ?', (id,joint_name))
    record = cursor.fetchone()
    if record:
        cursor.execute('''UPDATE angle_data SET min_angle = ?, max_angle = ?, timestamp=? WHERE id = ? AND joint_name=?''', 
                       (min_angle, max_angle, timestamp, id, joint_name))
    else:
        cursor.execute('''INSERT into angle_data(id, joint_name, min_angle, max_angle, timestamp) 
                       VALUES(?, ?, ?, ?, ?)''', (id, joint_name, min_angle, max_angle, timestamp))
          
def calculate_angle_points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def calculate_extension(shoulder, wrist):
    vertical_line = [shoulder[0], shoulder[1] + 100]
    limb_angle = calculate_angle_points(wrist, shoulder, vertical_line)

    # Extension occurs when the angle is less than 180°
    if limb_angle < 180:
        return 180 - limb_angle
    return 0

def calculate_flex_extension(shoulder, wrist):
    vertical_line = [shoulder[0], shoulder[1] + 100]
    limb_angle = calculate_angle_points(wrist, shoulder, vertical_line)

    return 180 - limb_angle

def calculate_angle_ref(reference, target):
    reference = np.array(reference)
    target = np.array(target)
    dot_product = np.dot(reference, target)
    magnitude_reference = np.linalg.norm(reference)
    magnitude_target = np.linalg.norm(target)
    cosine_angle = dot_product / (magnitude_reference * magnitude_target + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_lateral(results, cursor):
    """Process the frame to calculate and update neck angles."""
    landmarks = results.pose_landmarks.landmark
    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    nose = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
    neck = [(shoulder_left[0] + shoulder_right[0]) / 2, (shoulder_left[1] + shoulder_right[1]) / 2]

    neck_angles = {
        'C.Lateral FLX (RT)': 90 - calculate_angle_points(nose, neck, shoulder_right),
        'C.Lateral FLX (LT)': 90 - calculate_angle_points(nose, neck, shoulder_left)
    }

    # Apply the constraint: Angle should not exceed 22
    for name in neck_angles.keys():
        neck_angles[name] = min(neck_angles[name], 22)

    for name, angle in neck_angles.items():
        max_angle = max(cursor.execute('SELECT IFNULL(MAX(max_angle), 0) FROM angle_data WHERE joint_name = ? AND id = ?', (name,id, )).fetchone()[0], angle)
        min_angle = 0
        update_angle_data(cursor, name, min_angle, max_angle)
    
    return neck_angles

def display_angles(blank_frame, neck_angles, max_min_angles):
    """Overlay neck angles on the frame."""
    for name, angle in neck_angles.items():
        cv2.putText(blank_frame, f"{name}: {abs(int(angle))}", (10, 30 + 30 * list(neck_angles.keys()).index(name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # Display max/min angles
    y_position = 100
    for name, values in max_min_angles.items():
        cv2.putText(blank_frame, f"{name} Max: {abs(int(values['max']))}", (10, y_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        y_position += 25

def track_elbow_left(frame, conn, cursor, displayMode="blank"): 
    #max_angle = 0  # To track the maximum angle
    #min_angle = 180

    cursor.execute('SELECT min_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Left Elbow"))
    sql_min_angle = cursor.fetchone()
    min_angle = int(sql_min_angle[0]) if sql_min_angle else 180

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Left Elbow"))
    sql_max_angle = cursor.fetchone()
    max_angle = int(sql_max_angle[0]) if sql_max_angle else 0
    

    # Flip the frame horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(frame_rgb)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    # Get pose landmarks
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Manually copy landmarks to remap left and right due to flipping
        flipped_landmarks = [landmarks[i] for i in range(len(landmarks))]
        LEFT_RIGHT_MAPPING = {
            mp_pose.PoseLandmark.LEFT_SHOULDER.value: mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value: mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value: mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value: mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value: mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value: mp_pose.PoseLandmark.LEFT_WRIST.value,
        }

        for left, right in LEFT_RIGHT_MAPPING.items():
            flipped_landmarks[left], flipped_landmarks[right] = (
                landmarks[right],
                landmarks[left],
            )

        # Use flipped landmarks for calculations
        shoulder = [flipped_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    flipped_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [flipped_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    flipped_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [flipped_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    flipped_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate the original angle
        elbow_angle = 180 - calculate_angle_points(shoulder, elbow, wrist)

        # Angle should be between 0 and 150 degrees
        if elbow_angle > 150:
            elbow_angle = 150

        # Update the maximum and minimum angle
        max_angle = max(max_angle, elbow_angle)
        min_angle = min(min_angle, elbow_angle)
        update_angle_data(cursor, "Left Elbow", min_angle, max_angle)
        conn.commit()


        # Display text on the flipped frame
        cv2.putText(disFrame, f'Left Elbow Flexion: {int(elbow_angle)}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(disFrame, f'Min Flexion: {int(min_angle)}', (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disFrame, f'Max Flexion: {int(max_angle)}', (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(disFrame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        # Encode and send the frame
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_elbow_right(frame, conn, cursor, displayMode="blank"):
   
    #max_angle = 0  # To track the maximum angle
    #min_angle = 180 
    # 
    cursor.execute('SELECT min_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Right Elbow"))
    sql_min_angle = cursor.fetchone()
    min_angle = int(sql_min_angle[0]) if sql_min_angle else 180

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Right Elbow"))
    sql_max_angle = cursor.fetchone()
    max_angle = int(sql_max_angle[0]) if sql_max_angle else 0

    # Flip the frame horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(frame_rgb)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    # Get pose landmarks
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Remap landmarks due to flipping (left ↔ right)
        flipped_landmarks = [landmarks[i] for i in range(len(landmarks))]
        LEFT_RIGHT_MAPPING = {
            mp_pose.PoseLandmark.LEFT_SHOULDER.value: mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value: mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value: mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value: mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value: mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value: mp_pose.PoseLandmark.LEFT_WRIST.value,
        }

        for left, right in LEFT_RIGHT_MAPPING.items():
            flipped_landmarks[left], flipped_landmarks[right] = (
                landmarks[right],
                landmarks[left],
            )

        # Use remapped landmarks for calculations (right elbow)
        shoulder = [flipped_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    flipped_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [flipped_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    flipped_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [flipped_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    flipped_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate the angle
        angle = 180 - calculate_angle_points(shoulder, elbow, wrist)

        # Angle should be between 0 and 150 degrees
        if angle > 150:
            angle = 150

        # Update the maximum and minimum angle
        max_angle = max(max_angle, angle)
        min_angle = min(min_angle, angle)
        update_angle_data(cursor, "Right Elbow", min_angle, max_angle)
        conn.commit()


        # Display text on the flipped frame
        cv2.putText(disFrame, f'Right Elbow Flexion: {angle:.2f}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(disFrame, f'Min Flexion: {min_angle:.2f}', (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disFrame, f'Max Flexion: {max_angle:.2f}', (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(disFrame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        # Encode and send the frame
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_extension_left(frame, conn, cursor, displayMode="blank"):
    #max_extension = 0  # Maximum forward angle
    #min_extension = float("inf")  # Minimum forward angle (initialize with infinity)

    cursor.execute('SELECT min_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Shoulder Left Extension"))
    sql_min_angle = cursor.fetchone()
    min_extension = int(sql_min_angle[0]) if sql_min_angle else float("inf")

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Shoulder Left Extension"))
    sql_max_angle = cursor.fetchone()
    max_extension = int(sql_max_angle[0]) if sql_max_angle else 0


    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Create a copy of landmarks and remap for flipping
        flipped_landmarks = [landmarks[i] for i in range(len(landmarks))]
        LEFT_RIGHT_MAPPING = {
            mp_pose.PoseLandmark.LEFT_SHOULDER.value: mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value: mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value: mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value: mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value: mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value: mp_pose.PoseLandmark.LEFT_WRIST.value,
        }

        for left, right in LEFT_RIGHT_MAPPING.items():
            flipped_landmarks[left], flipped_landmarks[right] = (
                landmarks[right],
                landmarks[left],
            )

        # Extract frame dimensions for scaling landmarks
        frame_height, frame_width, _ = frame.shape

        # Get coordinates for the left shoulder and left wrist (after remapping)
        shoulder = [flipped_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width,
                    flipped_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height]
        wrist = [flipped_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame_width,
                    flipped_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame_height]

        # Calculate extension
        extension = 180 - calculate_extension(shoulder, wrist)

        # Limit angle range between 0 and 60 degrees
        if extension > 60:
            extension = 60

        # Update maximum and minimum values
        max_extension = max(max_extension, extension)
        min_extension = min(min_extension, extension)

        # Update the database with min and max values
        update_angle_data(cursor, "Shoulder Left Extension", min_extension, max_extension)
        conn.commit()


        # Display angles and max extension
        cv2.putText(disFrame, f"Extension: {extension:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Extension: {max_extension:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw key lines for visualization
        cv2.line(disFrame, tuple(map(int, shoulder)), (int(shoulder[0]), int(shoulder[1] + 100)), (0, 255, 0), 2)  # Vertical body line
        cv2.line(disFrame, tuple(map(int, shoulder)), tuple(map(int, wrist)), (255, 0, 0), 2)  # Arm line

        # Draw key points
        cv2.circle(disFrame, tuple(map(int, shoulder)), 5, (0, 255, 0), -1)
        cv2.circle(disFrame, tuple(map(int, wrist)), 5, (255, 0, 0), -1)


        # Encode the frame and yield for streaming
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_extension_right(frame, conn,cursor, displayMode="blank"):

    #max_extension = 0  # Maximum forward angle
    #min_extension = float("inf")  # Minimum forward angle (initialize with infinity)

    cursor.execute('SELECT min_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Shoulder Right Extension"))
    sql_min_angle = cursor.fetchone()
    min_extension = int(sql_min_angle[0]) if sql_min_angle else float("inf")

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Shoulder Right Extension"))
    sql_max_angle = cursor.fetchone()
    max_extension = int(sql_max_angle[0]) if sql_max_angle else 0

    frame = cv2.flip(frame, 1)        

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Create a copy of landmarks and remap for flipping
        flipped_landmarks = [landmarks[i] for i in range(len(landmarks))]
        LEFT_RIGHT_MAPPING = {
            mp_pose.PoseLandmark.LEFT_SHOULDER.value: mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value: mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value: mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value: mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value: mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value: mp_pose.PoseLandmark.LEFT_WRIST.value,
        }

        for left, right in LEFT_RIGHT_MAPPING.items():
            flipped_landmarks[left], flipped_landmarks[right] = (
                landmarks[right],
                landmarks[left],
            )

        # Extract frame dimensions for scaling landmarks
        frame_height, frame_width, _ = frame.shape

        # Get coordinates for shoulder and wrist (right side as example)
        shoulder = [flipped_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                    flipped_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
        wrist = [flipped_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width,
                    flipped_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height]

        # Calculate extension
        extension = 180 - calculate_extension(shoulder, wrist)

        #Angle should be between 0 and 160 degrees
        if extension > 60:
            extension = 60

        # Update maximum and minimum values
        max_extension = max(max_extension, extension)
        min_extension = min(min_extension, extension)

        update_angle_data(cursor,"Shoulder Right Extension", min_extension, max_extension)
        conn.commit()
        

        # Display angles and max extension
        cv2.putText(disFrame, f"Extension: {extension:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Extension: {max_extension:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw key lines for visualization
        cv2.line(disFrame, tuple(map(int, shoulder)), (int(shoulder[0]), int(shoulder[1] + 100)), (0, 255, 0), 2)  # Vertical body line
        cv2.line(disFrame, tuple(map(int, shoulder)), tuple(map(int, wrist)), (255, 0, 0), 2)  # Arm line

        # Draw key points
        cv2.circle(disFrame, tuple(map(int, shoulder)), 5, (0, 255, 0), -1)
        cv2.circle(disFrame, tuple(map(int, wrist)), 5, (255, 0, 0), -1)

    
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_flexion_left(frame, conn,cursor, displayMode="blank"):

    #max_flexion = 0  # Maximum forward angle
    #min_flexion = float("inf")  # Minimum forward angle (initialize with infinity)

    cursor.execute('SELECT min_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Shoulder Left Flexion"))
    sql_min_angle = cursor.fetchone()
    min_flexion = int(sql_min_angle[0]) if sql_min_angle else float("inf")

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Shoulder Left Flexion"))
    sql_max_angle = cursor.fetchone()
    max_flexion = int(sql_max_angle[0]) if sql_max_angle else 0

    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Create a copy of landmarks and remap for flipping
        flipped_landmarks = [landmarks[i] for i in range(len(landmarks))]
        LEFT_RIGHT_MAPPING = {
            mp_pose.PoseLandmark.LEFT_SHOULDER.value: mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value: mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value: mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value: mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value: mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value: mp_pose.PoseLandmark.LEFT_WRIST.value,
        }

        for left, right in LEFT_RIGHT_MAPPING.items():
            flipped_landmarks[left], flipped_landmarks[right] = (
                landmarks[right],
                landmarks[left],
            )

        # Extract frame dimensions for scaling landmarks
        frame_height, frame_width, _ = frame.shape

        # Get coordinates for shoulder and wrist (right side as example)
        shoulder = [flipped_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                    flipped_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
        wrist = [flipped_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame_width,
                    flipped_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame_height]

        # Calculate flexion
        flexion = 180 - calculate_flex_extension(shoulder, wrist)

        #Angle should be between 0 and 160 degrees
        if flexion > 180:
            flexion = 180

        # Update maximum and minimum values
        max_flexion = max(max_flexion, flexion)
        min_flexion = min(min_flexion, flexion)

        update_angle_data(cursor,"Shoulder Left Flexion", min_flexion, max_flexion)
        conn.commit()
        

        # Display angles and max flexion
        cv2.putText(disFrame, f"Flexion: {flexion:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Flexion: {max_flexion:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw key lines for visualization
        cv2.line(disFrame, tuple(map(int, shoulder)), (int(shoulder[0]), int(shoulder[1] + 100)), (0, 255, 0), 2)  # Vertical body line
        cv2.line(disFrame, tuple(map(int, shoulder)), tuple(map(int, wrist)), (255, 0, 0), 2)  # Arm line

        # Draw key points
        cv2.circle(disFrame, tuple(map(int, shoulder)), 5, (0, 255, 0), -1)
        cv2.circle(disFrame, tuple(map(int, wrist)), 5, (255, 0, 0), -1)

    
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_flexion_right(frame, conn,cursor, displayMode="blank"):

    #max_flexion = 0  # Maximum forward angle
    #min_flexion = float("inf")  # Minimum forward angle (initialize with infinity)

    cursor.execute('SELECT min_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Shoulder Right Flexion"))
    sql_min_angle = cursor.fetchone()
    min_flexion = int(sql_min_angle[0]) if sql_min_angle else float("inf")

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Shoulder Right Flexion"))
    sql_max_angle = cursor.fetchone()
    max_flexion = int(sql_max_angle[0]) if sql_max_angle else 0

    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        landmarks = results.pose_landmarks.landmark

        # Create a copy of landmarks and remap for flipping
        flipped_landmarks = [landmarks[i] for i in range(len(landmarks))]
        LEFT_RIGHT_MAPPING = {
            mp_pose.PoseLandmark.LEFT_SHOULDER.value: mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value: mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_ELBOW.value: mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            mp_pose.PoseLandmark.RIGHT_ELBOW.value: mp_pose.PoseLandmark.LEFT_ELBOW.value,
            mp_pose.PoseLandmark.LEFT_WRIST.value: mp_pose.PoseLandmark.RIGHT_WRIST.value,
            mp_pose.PoseLandmark.RIGHT_WRIST.value: mp_pose.PoseLandmark.LEFT_WRIST.value,
        }

        for left, right in LEFT_RIGHT_MAPPING.items():
            flipped_landmarks[left], flipped_landmarks[right] = (
                landmarks[right],
                landmarks[left],
            )

        # Extract frame dimensions for scaling landmarks
        frame_height, frame_width, _ = frame.shape

        # Get coordinates for shoulder and wrist (right side as example)
        shoulder = [flipped_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame_width,
                    flipped_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height]
        wrist = [flipped_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width,
                    flipped_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height]

        # Calculate flexion
        flexion = 180 - calculate_flex_extension(shoulder, wrist)

        #Angle should be between 0 and 160 degrees
        if flexion > 180:
            flexion = 180

        # Update maximum and minimum values
        max_flexion = max(max_flexion, flexion)
        min_flexion = min(min_flexion, flexion)

        update_angle_data(cursor,"Shoulder Right Flexion", min_flexion, max_flexion)
        conn.commit()

        # Display angles and max flexion
        cv2.putText(disFrame, f"Flexion: {flexion:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Flexion: {max_flexion:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw key lines for visualization
        cv2.line(disFrame, tuple(map(int, shoulder)), (int(shoulder[0]), int(shoulder[1] + 100)), (0, 255, 0), 2)  # Vertical body line
        cv2.line(disFrame, tuple(map(int, shoulder)), tuple(map(int, wrist)), (255, 0, 0), 2)  # Arm line

        # Draw key points
        cv2.circle(disFrame, tuple(map(int, shoulder)), 5, (0, 255, 0), -1)
        cv2.circle(disFrame, tuple(map(int, wrist)), 5, (255, 0, 0), -1)

    
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_cervical_flexion(frame, conn, cursor,displayMode="blank"):

    #min_flexion = 0  # Minimum angle for flexion

    global id

    cursor.execute('SELECT joint_name FROM angle_data WHERE id = ? AND joint_name = ?', (id-1, "C.FLX"))
    prev_joint = cursor.fetchone()

    if prev_joint == 'C.FLX':
        if id>=0 :
            id-=1
        else:
            id=0
        min_flexion = 0
    else:
        cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "C.FLX"))
        sql_angle = cursor.fetchone()
        min_flexion = int(sql_angle[0]) if sql_angle else 0

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results_pose = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results_pose.pose_landmarks:
        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Extract key landmarks
        landmarks = results_pose.pose_landmarks.landmark
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * frame_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * frame_height]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]

        # Compute the ear-to-shoulder line
        ear_to_shoulder_line = [left_shoulder[0] - left_ear[0], left_shoulder[1] - left_ear[1]]

        # Vertical reference vector
        vertical_reference = [0, -1]  # Vertical line in screen coordinates

        # Calculate angle with respect to the vertical line
        angle = abs(calculate_angle_ref(vertical_reference, ear_to_shoulder_line) - 180 )

        if angle>=80:
            angle = 80

        # Apply constraints to the ranges
        min_flexion = max(min_flexion, angle)

        # Save the angles to the database
        update_angle_data(cursor, "C.FLX", 0, abs(min_flexion))
        conn.commit()

        
        # Draw the ear-to-shoulder line and vertical reference
        cv2.line(disFrame, tuple(map(int, left_shoulder)), tuple(map(int, left_ear)), (255, 0, 0), 2)  # Ear-to-shoulder line
        cv2.line(disFrame, tuple(map(int, left_shoulder)), (int(left_shoulder[0]), int(left_shoulder[1]+150)), (0, 255, 0), 2)  # Vertical reference

        # Draw key landmarks
        cv2.circle(disFrame, tuple(map(int, left_ear)), 5, (0, 255, 0), -1)  # Right ear
        cv2.circle(disFrame, tuple(map(int, left_shoulder)), 5, (255, 0, 0), -1)  # Right shoulder

        #frame = cv2.flip(frame, 1)

        # Display information on the frame
        cv2.putText(disFrame, f"Current Angle: {abs(angle):.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(disFrame, f"Flexion: {abs(min_flexion):.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

   
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_cervical_extension(frame, conn,cursor,displayMode="blank"):

    # Tracking variables
    #max_extension = 0  # Maximum angle for reflexion

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "C.EXT"))
    sql_angle = cursor.fetchone()
    max_extension = int(sql_angle[0]) if sql_angle else 0
    #print(sql_angle)

    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results_pose = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results_pose.pose_landmarks:
        # Get frame dimensions
        frame_height, frame_width, _ = frame.shape

        # Extract key landmarks
        landmarks = results_pose.pose_landmarks.landmark
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * frame_width,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * frame_height]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]

        # Compute the ear-to-shoulder line
        ear_to_shoulder_line = [left_shoulder[0] - left_ear[0], left_shoulder[1] - left_ear[1]]

        # Vertical reference vector
        vertical_reference = [0, -1]  # Vertical line in screen coordinates

        # Calculate angle with respect to the vertical line
        angle = abs(calculate_angle_ref(vertical_reference, ear_to_shoulder_line) - 180)

        #if angle<=

        # Apply constraints to the ranges
        max_extension = max(max_extension, angle)

        # Save the angles to the database
        update_angle_data(cursor, "C.EXT", 0, abs(max_extension))
        conn.commit()


        # Draw the ear-to-shoulder line and vertical reference
        cv2.line(disFrame, tuple(map(int, left_shoulder)), tuple(map(int, left_ear)), (255, 0, 0), 2)  # Ear-to-shoulder line
        cv2.line(disFrame, tuple(map(int, left_shoulder)), (int(left_shoulder[0]), int(left_shoulder[1]+150)), (0, 255, 0), 2)  # Vertical reference

        # Draw key landmarks
        cv2.circle(disFrame, tuple(map(int, left_ear)), 5, (0, 255, 0), -1)  # Right ear
        cv2.circle(disFrame, tuple(map(int, left_shoulder)), 5, (255, 0, 0), -1)  # Right shoulder

        

        # Display information on the disFrame
        cv2.putText(disFrame, f"Current Angle: {abs(angle):.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(disFrame, f"Extension: {abs(max_extension):.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_cervical_lateral(frame, conn,cursor, displayMode="blank"):

    global id
    
    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "C.Lateral FLX (RT)"))
    sql_rt_angle = cursor.fetchone()
    max_right_deviation = int(sql_rt_angle[0]) if sql_rt_angle else 0

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "C.Lateral FLX (LT)"))
    sql_lt_angle = cursor.fetchone()
    max_left_deviation = int(sql_lt_angle[0]) if sql_lt_angle else 0

    # Process image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    blank_frame = np.zeros_like(image)
    blank_frame = cv2.flip(blank_frame, 1)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    # Process frame and calculate angles
    if results.pose_landmarks:
        neck_angles = process_lateral(results, cursor)
        conn.commit()
        
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



        # Draw the shoulder line
        cv2.line(disFrame, tuple(map(int, left_shoulder)), tuple(map(int, right_shoulder)), (0, 255, 0), 2)

        # Draw the perpendicular line to the nose
        cv2.line(disFrame, tuple(map(int, mid_shoulder)), tuple(map(int, nose)), (0, 255, 255), 2)

        # Draw points
        cv2.circle(disFrame, tuple(map(int, left_shoulder)), 5, (0, 0, 255), -1)
        cv2.circle(disFrame, tuple(map(int, right_shoulder)), 5, (0, 0, 255), -1)
        cv2.circle(disFrame, tuple(map(int, mid_shoulder)), 5, (0, 255, 0), -1)
        cv2.circle(disFrame, tuple(map(int, nose)), 5, (0, 255, 255), -1)

        # Display current deviations and max deviations
        cv2.putText(disFrame, f"Left Lateral: {neck_angles['C.Lateral FLX (LT)']:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Right Lateral: {neck_angles['C.Lateral FLX (RT)']:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disFrame, f"Max Left: {max_left_deviation:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Right: {max_right_deviation:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        #display_angles(disFrame, neck_angles, display_max_min_angles)
        mp_drawing.draw_landmarks(disFrame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_cervical_rotation(frame, conn,cursor, displayMode="blank"):

    #max_left_deviation = 0  # Track max deviation for the left
    #max_right_deviation = 0  # Track max deviation for the right

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "C.Rotation (RT)"))
    sql_rt_angle = cursor.fetchone()
    max_right_deviation = int(sql_rt_angle[0]) if sql_rt_angle else 0

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "C.Rotation (LT)"))
    sql_lt_angle = cursor.fetchone()
    max_left_deviation = int(sql_lt_angle[0]) if sql_lt_angle else 0

    
    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    blank_frame = cv2.flip(blank_frame, 1)
    results = pose.process(rgb_frame)

    disFrame = frame if displayMode == "normal" else blank_frame

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
        left_deviation = calculate_angle_points(nose, mid_shoulder, right_shoulder) - 90
        right_deviation = calculate_angle_points(nose, mid_shoulder, left_shoulder) - 90

        #left_deviation should be between 0 and 80 degrees
        if left_deviation > 90:
            left_deviation = 90

        #right_deviation should be between 0 and 80 degrees
        if right_deviation > 90:
            right_deviation = 90

        # Track max deviations independently
        max_left_deviation = max(max_left_deviation, left_deviation)
        max_right_deviation = max(max_right_deviation, right_deviation)


        # Draw the shoulder line
        cv2.line(disFrame, tuple(map(int, left_shoulder)), tuple(map(int, right_shoulder)), (0, 255, 0), 2)

        # Draw the perpendicular line to the nose
        cv2.line(disFrame, tuple(map(int, mid_shoulder)), tuple(map(int, nose)), (0, 255, 255), 2)

        # Draw points
        cv2.circle(disFrame, tuple(map(int, left_shoulder)), 5, (0, 0, 255), -1)
        cv2.circle(disFrame, tuple(map(int, right_shoulder)), 5, (0, 0, 255), -1)
        cv2.circle(disFrame, tuple(map(int, mid_shoulder)), 5, (0, 255, 0), -1)
        cv2.circle(disFrame, tuple(map(int, nose)), 5, (0, 255, 255), -1)

        # Display current deviations and max deviations
        cv2.putText(disFrame, f"Left Rotation: {left_deviation:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Right Rotation: {right_deviation:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disFrame, f"Max Left: {max_left_deviation:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Right: {max_right_deviation:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    
        update_angle_data(cursor,"C.Rotation (LT)", 0, max_left_deviation)
        conn.commit()
        update_angle_data(cursor,"C.Rotation (RT)", 0, max_right_deviation)
        conn.commit()
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_axial_joint(frame, conn, cursor, displayMode="blank"):
    #max_upward_deviation = 0  # Track max upward deviation
    #max_downward_deviation = 0  # Track max downward deviation

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Axial Deviation (Upward"))
    sql_up_angle = cursor.fetchone()
    max_upward_deviation = int(sql_up_angle[0]) if sql_up_angle else 0

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "Axial Deviation (Downward"))
    sql_down_angle = cursor.fetchone()
    max_downward_deviation = int(sql_down_angle[0]) if sql_down_angle else 0


    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract frame dimensions for scaling landmarks
        frame_height, frame_width, _ = frame.shape

        # Get coordinates for the right ear and right eye
        left_eye = [
            landmarks[mp_pose.PoseLandmark.LEFT_EYE].x * frame_width,
            landmarks[mp_pose.PoseLandmark.LEFT_EYE].y * frame_height,
        ]
        left_ear = [
            landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * frame_width,
            landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * frame_height,
        ]

        # Adjust right_ear for reference line alignment
        left_ear[0] -= 10  # Offset for better reference line placement

        # Create a horizontal reference line starting from the right ear
        horizontal_line = [left_ear[0] - 100, left_ear[1]]

        # Calculate the angle between the ear-eye line and the horizontal line
        angle = calculate_angle_points(left_ear, left_eye, horizontal_line)

        # Adjust the angle for better calibration
        calibrated_angle = angle - 15

        # Determine movement direction and track deviations
        if calibrated_angle > 0:  # Upward movement
            max_upward_deviation = max(max_upward_deviation, calibrated_angle)
        else:  # Downward movement
            max_downward_deviation = max(max_downward_deviation, abs(calibrated_angle))


        # Display the same points and lines on the original frame
        cv2.circle(disFrame, tuple(map(int, left_ear)), 5, (0, 0, 255), -1)
        cv2.circle(disFrame, tuple(map(int, left_eye)), 5, (0, 255, 0), -1)
        cv2.line(disFrame, tuple(map(int, left_ear)), tuple(map(int, left_eye)), (0, 255, 0), 2)
        cv2.line(disFrame, tuple(map(int, left_ear)), tuple(map(int, horizontal_line)), (255, 0, 0), 2)

        disFrame = cv2.flip(disFrame, 1)

        # Display angle and max deviations on the original disFrame
        cv2.putText(disFrame, f"Axial Deviation: {calibrated_angle:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Upward: {max_upward_deviation:.2f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disFrame, f"Max Downward: {max_downward_deviation:.2f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


  
        # Update angle data in the database
        update_angle_data(cursor, "Axial Deviation (Upward)", 0, max_upward_deviation)
        update_angle_data(cursor, "Axial Deviation (Downward)", 0, max_downward_deviation)
        conn.commit()  
        
        # Encode and yield the frame
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_lumbar_lateral_right(frame, conn, cursor, displayMode="blank"):
    #max_right_flexion = 0  # Maximum right lateral flexion

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "T.Lumbar Lateral FLX (RT)"))
    sql_angle = cursor.fetchone()
    max_right_flexion = int(sql_angle[0]) if sql_angle else 0

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results.pose_landmarks:
        disFrame = cv2.flip(disFrame, 1)  # Flip the frame horizontally
        landmarks = results.pose_landmarks.landmark

        # Extract frame dimensions for scaling landmarks
        frame_height, frame_width, _ = frame.shape

        # Extract and swap coordinates for flipped frame correction
        try:
            right_hip = [
                (1 - landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) * frame_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame_height
            ]
            right_shoulder = [
                (1 - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) * frame_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame_height
            ]
        except IndexError:
            print("Error accessing landmarks.")

        # Ensure points are accurate
        right_hip = np.array(right_hip, dtype=np.float32)
        right_shoulder = np.array(right_shoulder, dtype=np.float32)

        # Define static vertical reference lines for both sides
        vertical_right = np.array([right_hip, [right_hip[0], right_hip[1] - 100]], dtype=np.float32)

        # Calculate deviation angles for left and right sides
        right_deviation = calculate_angle_ref(vertical_right[1] - vertical_right[0], right_shoulder - right_hip)

        # Track maximum deviations
        max_right_flexion = max(max_right_flexion, right_deviation)

        
        # Draw landmarks and lines for visualization
        cv2.circle(disFrame, tuple(right_hip.astype(int)), 8, (0, 0, 255), -1)  # Left hip
        cv2.circle(disFrame, tuple(right_shoulder.astype(int)), 8, (0, 255, 0), -1)  # Left shoulder

        # Draw lines connecting hips to shoulders
        cv2.line(disFrame, tuple(right_hip.astype(int)), tuple(right_shoulder.astype(int)), (0, 255, 0), 2)

        # Draw vertical reference lines
        cv2.line(disFrame, tuple(right_hip.astype(int)), tuple(vertical_right[1].astype(int)), (0, 255, 255), 2)

        disFrame = cv2.flip(disFrame, 1)  # Flip the disFrame horizontally

        
        # Display deviation angles and maximum values
        cv2.putText(disFrame, f"Right Flexion: {right_deviation:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Right Flexion: {max_right_flexion:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Update the database with max flexion values
        update_angle_data(cursor, "T.Lumbar Lateral FLX (RT)", 0, max_right_flexion)
        conn.commit()

        # Encode and yield the disFrame for streaming
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_lumbar_lateral_left(frame, conn, cursor, displayMode="blank"): 
    #max_left_flexion = 0  # Maximum left lateral flexion

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "T.Lumbar Lateral FLX (LT)"))
    sql_angle = cursor.fetchone()
    max_left_flexion = int(sql_angle[0]) if sql_angle else 0

    
    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results.pose_landmarks:
        disFrame = cv2.flip(disFrame, 1)
        landmarks = results.pose_landmarks.landmark

        # Extract frame dimensions for scaling landmarks
        frame_height, frame_width, _ = frame.shape

        # Extract and swap coordinates for flipped frame correction
        try:
            left_hip = [
                (1 - landmarks[mp_pose.PoseLandmark.LEFT_HIP].x) * frame_width,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height
            ]
            left_shoulder = [
                (1 - landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x) * frame_width,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height
            ]
            
        except IndexError:
            print("Error accessing landmarks.")

        # Ensure points are accurate
        left_hip = np.array(left_hip, dtype=np.float32)
        left_shoulder = np.array(left_shoulder, dtype=np.float32)

        # Define static vertical reference lines for both sides
        vertical_left = np.array([left_hip, [left_hip[0], left_hip[1] - 100]], dtype=np.float32)

        # Calculate deviation angles for left and right sides
        left_deviation = calculate_angle_ref(vertical_left[1] - vertical_left[0], left_shoulder - left_hip)

        # Track maximum deviations
        max_left_flexion = max(max_left_flexion, left_deviation)

        
        # Draw landmarks and lines for visualization
        cv2.circle(disFrame, tuple(left_hip.astype(int)), 8, (255, 0, 0), -1)  # Right hip
        cv2.circle(disFrame, tuple(left_shoulder.astype(int)), 8, (0, 255, 255), -1)  # Right shoulder

        # Draw lines connecting hips to shoulders
        cv2.line(disFrame, tuple(left_hip.astype(int)), tuple(left_shoulder.astype(int)), (255, 255, 0), 2)

        # Draw vertical reference lines
        cv2.line(disFrame, tuple(left_hip.astype(int)), tuple(vertical_left[1].astype(int)), (255, 0, 255), 2)

        disFrame = cv2.flip(disFrame, 1)
        # Display deviation angles and maximum values
        cv2.putText(disFrame, f"Left Flexion: {left_deviation:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disFrame, f"Max Left Flexion: {max_left_flexion:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        # Update the database with max flexion values
        update_angle_data(cursor, "T.Lumbar Lateral FLX (LT)", 0, max_left_flexion)
        conn.commit()
        # Encode and yield the disFrame for streaming
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')

def track_lumbar_flexion_extension(frame, conn, cursor, displayMode="blank"):
    #max_flexion = 0  # Maximum forward flexion
    #max_extension = 0  # Maximum backward extension

    global id

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "T.Lumbar FLX"))
    sql_flx_angle = cursor.fetchone()
    max_flexion = int(sql_flx_angle[0]) if sql_flx_angle else 0

    cursor.execute('SELECT max_angle FROM angle_data WHERE id = ? AND joint_name = ?', (id, "T.Lumbar EXT"))
    sql_ext_angle = cursor.fetchone()
    max_extension = int(sql_ext_angle[0]) if sql_ext_angle else 0

    # Convert to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blank_frame = np.zeros_like(frame)
    results = pose.process(rgb_frame)

    if displayMode == "normal":
        disFrame = frame
    else:
        disFrame = blank_frame

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract frame dimensions for scaling landmarks
        frame_height, frame_width, _ = frame.shape

        # Get coordinates for the left shoulder and left hip
        left_shoulder = [(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame_width),
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame_height]
        left_hip = [(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame_width),
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame_height]

        # Calculate the three-quarter point between shoulder and hip
        three_quarter_point = [
            left_shoulder[0] + 0.75 * (left_hip[0] - left_shoulder[0]),
            left_shoulder[1] + 0.75 * (left_hip[1] - left_shoulder[1])
        ]

        # Define the static vertical line (vertical reference)
        vertical_reference = [0, 1]  # Vertical unit vector for comparison

        # Dynamic line vector: shoulder to three-quarter point
        dynamic_vector = [
            three_quarter_point[0] - left_shoulder[0],
            three_quarter_point[1] - left_shoulder[1]
        ]

        # Calculate the angle between the vertical line and the dynamic line
        flexion_extension_angle = calculate_angle_ref(vertical_reference, dynamic_vector)

        # Determine flexion or extension
        if dynamic_vector[0] > 0:  # Forward from vertical (flexion)
            max_flexion = max(max_flexion, flexion_extension_angle)
        else:  # Backward from vertical (extension)
            max_extension = max(max_extension, flexion_extension_angle)

        if max_flexion>50:
            max_flexion=50
        if max_extension>25:
            max_extension=25
        
        # Draw the landmarks and lines
        cv2.circle(disFrame, tuple(map(int, left_shoulder)), 5, (0, 255, 0), -1)
        cv2.circle(disFrame, tuple(map(int, left_hip)), 3, (255, 0, 0), -1)
        cv2.circle(disFrame, tuple(map(int, three_quarter_point)), 5, (0, 0, 255), -1)

        # Draw the dynamic line (shoulder to three-quarter point)
        cv2.line(disFrame, tuple(map(int, left_shoulder)), tuple(map(int, three_quarter_point)), (0, 255, 255), 2)

        # Draw the static vertical reference line
        cv2.line(disFrame, tuple(map(int, left_shoulder)), (int(left_shoulder[0]), int(left_shoulder[1] + 100)), (255, 255, 0), 2)

        # Display the current angle and maximum values
        cv2.putText(disFrame, f"Flexion Angle: {flexion_extension_angle:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(disFrame, f"Max Flexion: {max_flexion:.2f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disFrame, f"Max Extension: {max_extension:.2f}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Update the database with max flexion and extension values
        update_angle_data(cursor, "T.Lumbar FLX", 0, max_flexion)
        update_angle_data(cursor, "T.Lumbar EXT", 0, max_extension)
        conn.commit()
        # Encode the disFrame and yield for streaming
        _, buffer = cv2.imencode('.jpg', disFrame)
        return base64.b64encode(buffer).decode('utf-8')


#Need to retrieve max angles from straightly db nearly 17 places.
#Standardize the colors for static and dynamic lines including text and points color.
#Set range for max angles

# Mapping function names to functions
TRACK_FUNCTIONS = {
    "Cervical_Flexion": (track_cervical_flexion, {"conn": None, "cursor": None}),
    "Cervical_Extension": (track_cervical_extension, {"conn": None, "cursor": None}),
    "Cervical_Lateral_Flexion": (track_cervical_lateral, {"conn": None, "cursor": None}),
    "Cervical_Rotation": (track_cervical_rotation, {"conn": None, "cursor": None}),
    "Thoraco_Lumbar_Lateral_Right": (track_lumbar_lateral_right, {"conn": None, "cursor": None}),
    "Thoraco_Lumbar_Lateral_Left": (track_lumbar_lateral_left, {"conn": None, "cursor": None}),
    "Thoraco_Lumbar_Flexion_Extension": (track_lumbar_flexion_extension, {"conn": None, "cursor": None}),    
}
OTHER_TRACK_FUNCTIONS = {
    "Elbow_right": (track_elbow_right, {"conn": None, "cursor": None}),
    "Elbow_Left": (track_elbow_left, {"conn": None, "cursor": None}),
    "Shoulder_Left_Extension": (track_extension_left, {"conn": None, "cursor": None}),
    "Shoulder_Right_Extension": (track_extension_right, {"conn": None, "cursor": None}),
    "Shoulder_Left_Flexion": (track_flexion_left, {"conn": None, "cursor": None}),
    "Shoulder_Right_Flexion": (track_flexion_right, {"conn": None, "cursor": None}),
    "Axial_Joint_movement": (track_axial_joint, {"conn": None, "cursor": None}),
}

# Combine track functions for easy management
ALL_TRACK_FUNCTIONS = {**TRACK_FUNCTIONS, **OTHER_TRACK_FUNCTIONS}

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/main")
def main_page():
    # Initial options to display on the main page
    options = list(TRACK_FUNCTIONS.keys())
    return render_template("main.html", options=options, more_available=True)

@app.route("/more-options")
def more_options():
    # Additional options from OTHER_TRACK_FUNCTIONS
    additional_options = list(OTHER_TRACK_FUNCTIONS.keys())
    return {"options": additional_options}

@app.route("/track/<function_name>")
def track(function_name):
    if function_name in ALL_TRACK_FUNCTIONS:
        if function_name == 'Cervical_Flexion' :
            global id
            id+=1
        return render_template("track.html", function_name=function_name)
    else:
        return "Invalid function name", 404
    

@app.route('/process_frame/<function_name>', methods=['POST'])
def process_frame(function_name):
    data = request.get_json()
    frame_data = data.get('frame')
    displayMode = data.get('display_mode', 'blank')

    if not frame_data:
        return jsonify({"error": "No frame data received"}), 400

    # Decode Base64 Image
    img_data = base64.b64decode(frame_data.split(',')[1])
    image = Image.open(BytesIO(img_data))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    func_info = ALL_TRACK_FUNCTIONS.get(function_name)
    if func_info:
        func, default_args = func_info
        conn, cursor = init_db()  # Initialize the database connection
        updated_args = {**default_args, "conn": conn, "cursor": cursor, "displayMode": displayMode}
        frame_set = func(image, **updated_args)
        return jsonify({"processed_frame": frame_set})
    else:
        return "Invalid function name"

@app.route("/video_feed/<function_name>/<display_mode>")
def video_feed(function_name, display_mode):
    func_info = ALL_TRACK_FUNCTIONS.get(function_name)
    if func_info:
        func, default_args = func_info
        conn, cursor = init_db()  # Initialize the database connection
        updated_args = {**default_args, "conn": conn, "cursor": cursor, "displayMode": display_mode}
        return Response(func(**updated_args), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid function name"

def fetch_all_tracking_data():
    conn,cursor = init_db()
    cursor.execute("SELECT id,joint_name, min_angle, max_angle, timestamp FROM angle_data")
    results = cursor.fetchall()  # Fetch all results
    conn.close()  # Close the connection
    return results

@app.route('/results')
def results():
    tracking_data = fetch_all_tracking_data()  # Get data from the database
    return render_template("results.html", tracking_data=tracking_data)

@app.route('/download_csv')
def download_csv():
    conn, cursor = init_db()
    cursor.execute("SELECT id,joint_name, min_angle, max_angle, timestamp FROM angle_data")
    data = cursor.fetchall()
    conn.close()

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp = timestamp.replace(" ", "_").replace("-", "_").replace(":", "_")
    csv_filename = f"tracking_data_{timestamp}.csv"
    csv_filepath = os.path.join(CSV_EXPORT_DIR, csv_filename)

    # Write to CSV file
    with open(csv_filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Joint Name", "Minimum Angle", "Maximum Angle", "Last Updated"])
        
        for idx, row in enumerate(data, start=1):
            writer.writerow([
                idx,  # ID
                row[0],  # Joint Name
                row[1],  # Minimum Angle
                row[2],  # Maximum Angle
                row[3]   # Timestamp
            ])

    # Return the CSV as a downloadable file
    return send_file(csv_filepath, as_attachment=True, download_name=csv_filename, mimetype="text/csv")

if __name__ == "__main__":
    app.run(debug=True)
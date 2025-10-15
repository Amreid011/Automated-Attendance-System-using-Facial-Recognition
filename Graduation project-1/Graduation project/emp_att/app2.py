from flask import Flask, flash, render_template, request, redirect, url_for, session, jsonify
import os
import cv2
import pandas as pd
import numpy as np
import face_recognition
from datetime import datetime
import pickle
from pathlib import Path
import threading
from queue import Queue
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Performance and Quality Tuning Parameters
# Camera resolution settings (will fall back to lower resolution if not supported)
CAMERA_WIDTH = 1920  # Increased for better long-range detection
CAMERA_HEIGHT = 1080  # Increased for better long-range detection
FALLBACK_WIDTH = 1280  # Increased fallback resolution
FALLBACK_HEIGHT = 720  # Increased fallback resolution

# Frame processing settings
FRAME_SKIP = 1  # Process every frame for better detection
DETECTION_RESIZE_SCALE = 0.75  # Increased from 0.5 for better long-range detection
UPSAMPLE = 1  # Increased from 0 to improve detection at longer ranges
FACE_DETECTION_MODEL = 'cnn'  # Changed to CNN for better accuracy at longer ranges

# UI settings
STATUS_BAR_HEIGHT = 40
STATUS_BAR_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)

# Face recognition settings
FACE_RECOGNITION_TOLERANCE = 0.6  # Increased tolerance for long-range recognition
FACE_RECOGNITION_BATCH_SIZE = 3  # Reduced batch size for more frequent processing
FACE_RECOGNITION_CACHE_SIZE = 100  # Cache size for face encodings

# Face recognition stability settings
FACE_RECOGNITION_HISTORY_SIZE = 3  # Reduced from 5 to be more responsive
MIN_CONFIDENCE_THRESHOLD = 0.45  # Reduced from 0.6 to be more lenient
MIN_CONSECUTIVE_MATCHES = 2  # Reduced from 3 to be more responsive
FACE_DETECTION_MIN_SIZE = 20  # Reduced from 30 to detect smaller faces at longer range

# Camera configuration for long-range detection
CAMERA_CONFIG = {
    'ip': os.getenv('CAMERA_IP', '192.168.1.4'),
    'username': os.getenv('CAMERA_USER', 'martin'),
    'password': os.getenv('CAMERA_PASS', 'martin'),
    'port': int(os.getenv('CAMERA_PORT', '554')),
    'stream': os.getenv('CAMERA_STREAM', 'stream1'),
    'use_local_camera_fallback': True,
    'local_camera_index': 0,
    'rtsp_transport': 'tcp',
    'buffer_size': 1024 * 1024 * 2,  # Increased buffer size
    'timeout': 10000,  # Increased timeout
    'reconnect_delay': 1,
    'max_reconnect_attempts': 3,
    'cv2_config': {
        cv2.CAP_PROP_BUFFERSIZE: 5,  # Increased buffer size
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'H264'),
        cv2.CAP_PROP_AUTOFOCUS: 1,  # Enable autofocus
        cv2.CAP_PROP_FOCUS: 0,  # Set focus to infinity for long-range
        cv2.CAP_PROP_ZOOM: 0,  # Reset zoom
        cv2.CAP_PROP_EXPOSURE: -7  # Adjust exposure for better long-range visibility
    }
}

# Suppress FFmpeg warnings
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|loglevel;quiet'

# Face match threshold for long-range recognition
FACE_MATCH_THRESHOLD = 0.5  # Adjusted for better long-range recognition

# File paths
EMPLOYEE_FILE = 'employees.xlsx'
ATTENDANCE_FILE = 'attendance.xlsx'
FACE_DB_FILE = 'face_db.pkl'
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = '1234'

# Global caching variables
KNOWN_ENCODINGS = []
KNOWN_IDS = []
KNOWN_NAMES = []
EMPLOYEE_MAP = {}
NEW_ATTENDANCE_ROWS = []
FACE_ENCODING_CACHE = {}  # Cache for face encodings

class FaceRecognitionSmoother:
    """Class to handle face recognition smoothing and stability"""
    def __init__(self):
        self.recognition_history = {}  # Dictionary to store recognition history
        self.confirmed_faces = {}  # Dictionary to store confirmed face recognitions
        self.face_tracking = {}  # Dictionary to track face positions
        
    def update(self, face_id, recognition_result, confidence, face_location):
        """Update recognition history and return smoothed result"""
        if face_id not in self.recognition_history:
            self.recognition_history[face_id] = []
            self.face_tracking[face_id] = face_location
            
        # Update face tracking
        if face_id in self.face_tracking:
            # Calculate face movement
            old_location = self.face_tracking[face_id]
            new_location = face_location
            movement = abs(new_location[0] - old_location[0]) + abs(new_location[1] - old_location[1])
            
            # If face moved significantly, reduce confidence
            if movement > 50:  # Threshold for significant movement
                confidence *= 0.8
                
        self.face_tracking[face_id] = face_location
            
        # Add new recognition result
        self.recognition_history[face_id].append((recognition_result, confidence))
        
        # Keep only recent history
        if len(self.recognition_history[face_id]) > FACE_RECOGNITION_HISTORY_SIZE:
            self.recognition_history[face_id].pop(0)
            
        # Calculate smoothed result
        if len(self.recognition_history[face_id]) >= MIN_CONSECUTIVE_MATCHES:
            recent_results = self.recognition_history[face_id][-MIN_CONSECUTIVE_MATCHES:]
            # Count occurrences of each result
            result_counts = {}
            for r, c in recent_results:
                if c >= MIN_CONFIDENCE_THRESHOLD:
                    result_counts[r] = result_counts.get(r, 0) + 1
            
            # If we have a majority result, use it
            if result_counts:
                best_result = max(result_counts.items(), key=lambda x: x[1])
                if best_result[1] >= MIN_CONSECUTIVE_MATCHES - 1:  # Allow one mismatch
                    self.confirmed_faces[face_id] = (best_result[0], confidence)
                    return best_result[0], confidence
                
        # If we have a confirmed face, use it with decay
        if face_id in self.confirmed_faces:
            confirmed_result, confirmed_confidence = self.confirmed_faces[face_id]
            if confidence >= MIN_CONFIDENCE_THRESHOLD * 0.7:  # More lenient tolerance
                return confirmed_result, confirmed_confidence
                
        return None, 0.0
        
    def reset(self, face_id):
        """Reset recognition history for a face"""
        if face_id in self.recognition_history:
            del self.recognition_history[face_id]
        if face_id in self.confirmed_faces:
            del self.confirmed_faces[face_id]
        if face_id in self.face_tracking:
            del self.face_tracking[face_id]

# Initialize the smoother
FACE_SMOOTHER = FaceRecognitionSmoother()

class CameraStream:
    """Threaded camera stream to prevent buffer lag"""
    def __init__(self, camera_id=0):
        self.stream = None
        self.frame = None
        self.stopped = False
        self.thread = None
        self.camera_id = camera_id
        self.lock = threading.Lock()
        self.actual_width = FALLBACK_WIDTH
        self.actual_height = FALLBACK_HEIGHT
        self.reconnect_attempts = 0
        
    def _get_rtsp_url(self):
        """Construct RTSP URL from configuration"""
        return f"rtsp://{CAMERA_CONFIG['username']}:{CAMERA_CONFIG['password']}@{CAMERA_CONFIG['ip']}:{CAMERA_CONFIG['port']}/{CAMERA_CONFIG['stream']}"
        
    def _connect(self):
        """Attempt to connect to IP camera or fall back to local camera"""
        try:
            if CAMERA_CONFIG['use_local_camera_fallback'] and self.reconnect_attempts >= CAMERA_CONFIG['max_reconnect_attempts']:
                print("Max reconnect attempts reached, falling back to local camera")
                self.stream = cv2.VideoCapture(self.camera_id)
            else:
                rtsp_url = self._get_rtsp_url()
                print(f"Attempting to connect to IP camera: {rtsp_url}")
                self.stream = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                
                # Apply OpenCV configuration for long-range detection
                for prop, value in CAMERA_CONFIG['cv2_config'].items():
                    self.stream.set(prop, value)
                
                # Additional settings for long-range detection
                self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
                self.stream.set(cv2.CAP_PROP_FOCUS, 0)  # Set focus to infinity
                self.stream.set(cv2.CAP_PROP_ZOOM, 0)  # Reset zoom
                self.stream.set(cv2.CAP_PROP_EXPOSURE, -7)  # Adjust exposure
            
            if not self.stream.isOpened():
                raise RuntimeError("Failed to open camera")
            
            # Try to set full HD resolution first
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            
            # Verify if the resolution was set successfully
            actual_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width < CAMERA_WIDTH or actual_height < CAMERA_HEIGHT:
                # Fall back to lower resolution
                print(f"Camera doesn't support {CAMERA_WIDTH}x{CAMERA_HEIGHT}, falling back to {FALLBACK_WIDTH}x{FALLBACK_HEIGHT}")
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FALLBACK_WIDTH)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FALLBACK_HEIGHT)
                self.actual_width = FALLBACK_WIDTH
                self.actual_height = FALLBACK_HEIGHT
            else:
                self.actual_width = actual_width
                self.actual_height = actual_height
                print(f"Camera resolution set to {actual_width}x{actual_height}")
            
            # Test if we can actually read a frame
            ret, frame = self.stream.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read frame from camera")
            
            self.reconnect_attempts = 0  # Reset reconnect attempts on successful connection
            return True
            
        except Exception as e:
            if self.stream is not None:
                self.stream.release()
            self.reconnect_attempts += 1
            print(f"Failed to connect to camera: {str(e)}")
            if self.reconnect_attempts < CAMERA_CONFIG['max_reconnect_attempts']:
                print(f"Retrying in {CAMERA_CONFIG['reconnect_delay']} seconds...")
                time.sleep(CAMERA_CONFIG['reconnect_delay'])
                return self._connect()
            raise RuntimeError(f"Failed to connect to camera after {CAMERA_CONFIG['max_reconnect_attempts']} attempts: {str(e)}")
        
    def start(self):
        """Start the thread to read frames from the video stream"""
        if not self._connect():
            raise RuntimeError("Failed to connect to camera")
            
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self
        
    def _update(self):
        """Keep looping infinitely until the thread is stopped"""
        while not self.stopped:
            try:
                if not self.stream.grab():
                    # If grab fails, try to reconnect
                    if not self._connect():
                        time.sleep(2)  # Wait before retry
                        continue
                    continue
                    
                ret, frame = self.stream.retrieve()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame.copy()  # Store a copy of the frame
            except Exception as e:
                print(f"Error in camera stream: {e}")
                time.sleep(2)  # Wait before retry
                if not self._connect():
                    continue
                    
    def read(self):
        """Return the most recent frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
        
    def stop(self):
        """Stop the thread and release resources"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
        if self.stream is not None:
            self.stream.release()

# Initialize global camera stream
STREAM = None

def load_or_create_face_db():
    """Load face encodings from pickle file or create new if not exists"""
    global KNOWN_ENCODINGS, KNOWN_IDS, KNOWN_NAMES
    
    if os.path.exists(FACE_DB_FILE):
        try:
            with open(FACE_DB_FILE, 'rb') as f:
                KNOWN_ENCODINGS, KNOWN_IDS, KNOWN_NAMES = pickle.load(f)
            print(f"Loaded {len(KNOWN_ENCODINGS)} face encodings from {FACE_DB_FILE}")
            return
        except Exception as e:
            print(f"Error loading face database: {e}")
    
    # If pickle doesn't exist or failed to load, create new database
    print("Creating new face database...")
    for folder in os.listdir('dataset'):
        folder_path = os.path.join('dataset', folder)
        if not os.path.isdir(folder_path):
            continue
            
        emp_id = folder.split('_')[0]
        emp_name = " ".join(folder.split('_')[1:])
        
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                if face_locations:
                    encoding = face_recognition.face_encodings(image, face_locations)[0]
                    KNOWN_ENCODINGS.append(encoding)
                    KNOWN_IDS.append(emp_id)
                    KNOWN_NAMES.append(emp_name)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    # Save to pickle file
    try:
        with open(FACE_DB_FILE, 'wb') as f:
            pickle.dump((KNOWN_ENCODINGS, KNOWN_IDS, KNOWN_NAMES), f)
        print(f"Saved {len(KNOWN_ENCODINGS)} face encodings to {FACE_DB_FILE}")
    except Exception as e:
        print(f"Error saving face database: {e}")

def load_employee_map():
    """Load employee data into memory at startup"""
    global EMPLOYEE_MAP
    if os.path.exists(EMPLOYEE_FILE):
        try:
            df = pd.read_excel(EMPLOYEE_FILE, engine='openpyxl')
            EMPLOYEE_MAP = {int(row['ID']): row['Name'] for _, row in df.iterrows()}
            print(f"Loaded {len(EMPLOYEE_MAP)} employees into memory")
        except Exception as e:
            print(f"Error loading employee data: {e}")
            EMPLOYEE_MAP = {}

def add_employee_face_encodings(emp_id, name, folder_path):
    """Add face encodings for a new employee to the database"""
    global KNOWN_ENCODINGS, KNOWN_IDS, KNOWN_NAMES
    
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                KNOWN_ENCODINGS.append(encoding)
                KNOWN_IDS.append(emp_id)
                KNOWN_NAMES.append(name)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save updated database
    try:
        with open(FACE_DB_FILE, 'wb') as f:
            pickle.dump((KNOWN_ENCODINGS, KNOWN_IDS, KNOWN_NAMES), f)
        print(f"Updated face database with new employee {name}")
    except Exception as e:
        print(f"Error updating face database: {e}")

# Initialize caches at module level
load_or_create_face_db()
load_employee_map()

def init_employee_file():
    """Initialize employee Excel file with proper structure"""
    try:
        # Create a new DataFrame with the required columns
        df = pd.DataFrame(columns=['ID', 'Name', 'Department'])
        # Save to Excel with explicit engine
        df.to_excel(EMPLOYEE_FILE, index=False, engine='openpyxl')
        print(f"Created new employee file: {EMPLOYEE_FILE}")
    except Exception as e:
        print(f"Error creating employee file: {e}")
        raise

def init_attendance_file():
    """Initialize attendance Excel file with proper structure"""
    try:
        # Create a new DataFrame with the required columns
        df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time', 'Leave Time'])
        # Save to Excel with explicit engine
        df.to_excel(ATTENDANCE_FILE, index=False, engine='openpyxl')
        print(f"Created new attendance file: {ATTENDANCE_FILE}")
    except Exception as e:
        print(f"Error creating attendance file: {e}")
        raise

def load_employee_data():
    """Safely load employee data from Excel file"""
    if not os.path.exists(EMPLOYEE_FILE):
        init_employee_file()
    try:
        return pd.read_excel(EMPLOYEE_FILE, engine='openpyxl')
    except Exception as e:
        print(f"Error loading employee data: {e}")
        # If file is corrupted, reinitialize it
        if os.path.exists(EMPLOYEE_FILE):
            os.remove(EMPLOYEE_FILE)
        init_employee_file()
        return pd.DataFrame(columns=['ID', 'Name', 'Department'])

def load_attendance_data():
    """Safely load attendance data from Excel file"""
    if not os.path.exists(ATTENDANCE_FILE):
        init_attendance_file()
    try:
        return pd.read_excel(ATTENDANCE_FILE, engine='openpyxl')
    except Exception as e:
        print(f"Error loading attendance data: {e}")
        # If file is corrupted, reinitialize it
        if os.path.exists(ATTENDANCE_FILE):
            os.remove(ATTENDANCE_FILE)
        init_attendance_file()
        return pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time', 'Leave Time'])

def add_employee(emp_id, name, department, position):
    """Add new employee and update the cached employee map"""
    global EMPLOYEE_MAP
    
    if emp_id in EMPLOYEE_MAP:
        return False
        
    # Add to Excel file
    if os.path.exists(EMPLOYEE_FILE):
        df = pd.read_excel(EMPLOYEE_FILE, engine='openpyxl')
    else:
        df = pd.DataFrame(columns=['ID', 'Name', 'Department'])
    
    new_data = {'ID': emp_id, 'Name': name, 'Department': department}
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_excel(EMPLOYEE_FILE, index=False, engine='openpyxl')
    
    # Update cached map
    EMPLOYEE_MAP[emp_id] = name
    return True


def capture_employee_photos(emp_id, name, num_photos=10):
    global STREAM
    folder_name = f"dataset/{emp_id}_{name.replace(' ', '_')}"
    os.makedirs(folder_name, exist_ok=True)
    
    try:
        # Initialize camera stream if not already running
        if STREAM is None:
            STREAM = CameraStream(0)  # Use default camera (0)
            STREAM.start()
        
        count = 0
        while count < num_photos:
            frame = STREAM.read()
            if frame is None:
                continue
                
            cv2.imshow('Capturing Photos - Press w to quit', frame)
            img_path = os.path.join(folder_name, f"{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # After capturing photos, add their encodings to the database
        add_employee_face_encodings(emp_id, name, folder_name)
        
    except Exception as e:
        flash(f"Error during photo capture: {str(e)}", "error")
    finally:
        cv2.destroyAllWindows()


def load_known_faces(dataset_path='dataset'):
    known_encodings, known_ids, known_names = [], [], []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        emp_id = folder.split('_')[0]
        emp_name = " ".join(folder.split('_')[1:])
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_encodings.append(encoding)
                known_ids.append(emp_id)
                known_names.append(emp_name)
    return known_encodings, known_ids, known_names


def register_attendance(emp_id, name):
    """Add attendance record to the batch queue instead of writing immediately"""
    time_now = datetime.now().strftime('%H:%M:%S')
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check if already in today's batch
    for row in NEW_ATTENDANCE_ROWS:
        if row['ID'] == emp_id and row['Date'] == today_date:
            flash(f"{name} has already recorded attendance today.", "info")
            return
    
    # Check if already in existing attendance file
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        if not df[(df['ID'] == emp_id) & (df['Date'] == today_date)].empty:
            flash(f"{name} has already recorded attendance today.", "info")
            return
    
    # Add to batch queue
    NEW_ATTENDANCE_ROWS.append({
        'ID': emp_id,
        'Name': name,
        'Date': today_date,
        'Attendance Time': time_now,
        'Leave Time': ''
    })
    flash(f"Attendance queued for {name} at {time_now}.", "success")


def register_leave(emp_id, name):
    """Add leave record to the batch queue instead of writing immediately"""
    time_now = datetime.now().strftime('%H:%M:%S')
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # First check existing attendance file for today's record
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_excel(ATTENDANCE_FILE)
        existing_record = df[(df['ID'] == emp_id) & (df['Date'] == today_date)]
        if not existing_record.empty:
            # Update leave time in existing record
            df.at[existing_record.index[0], 'Leave Time'] = time_now
            df.to_excel(ATTENDANCE_FILE, index=False)
            flash(f"Leave recorded for {name} at {time_now}.", "success")
            return
    
    # Check if in today's batch
    for row in NEW_ATTENDANCE_ROWS:
        if row['ID'] == emp_id and row['Date'] == today_date:
            row['Leave Time'] = time_now
            flash(f"Leave queued for {name} at {time_now}.", "success")
            return
    
    flash(f"No attendance record found for {name} today.", "error")


def write_batched_attendance():
    """Write all batched attendance records to the Excel file"""
    if not NEW_ATTENDANCE_ROWS:
        return
    
    try:
        # Read existing attendance records
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_excel(ATTENDANCE_FILE, engine='openpyxl')
        else:
            df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time', 'Leave Time'])
        
        # Append new records
        new_df = pd.DataFrame(NEW_ATTENDANCE_ROWS)
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Write back to file
        df.to_excel(ATTENDANCE_FILE, index=False, engine='openpyxl')
        print(f"Wrote {len(NEW_ATTENDANCE_ROWS)} attendance records to {ATTENDANCE_FILE}")
        
        # Clear the batch queue
        NEW_ATTENDANCE_ROWS.clear()
    except Exception as e:
        print(f"Error writing batched attendance: {e}")
        flash("Error saving attendance records.", "error")


def process_frame_for_detection(frame):
    """Process frame for face detection with optimized parameters"""
    if frame is None:
        return None, None, None
    
    # Calculate new dimensions based on resize scale
    height, width = frame.shape[:2]
    new_width = int(width * DETECTION_RESIZE_SCALE)
    new_height = int(height * DETECTION_RESIZE_SCALE)
    
    # Resize frame for detection
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces with optimized parameters
    face_locations = face_recognition.face_locations(
        rgb_frame,
        model=FACE_DETECTION_MODEL,
        number_of_times_to_upsample=UPSAMPLE
    )
    
    if not face_locations:
        return frame, [], []
    
    # Scale face locations back to original size
    scale_factor = 1 / DETECTION_RESIZE_SCALE
    scaled_locations = []
    for (top, right, bottom, left) in face_locations:
        scaled_locations.append((
            int(top * scale_factor),
            int(right * scale_factor),
            int(bottom * scale_factor),
            int(left * scale_factor)
        ))
    
    # Get face encodings from the resized frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    return frame, scaled_locations, face_encodings

def recognize_face_and_register(mode='attend'):
    """Recognize faces and register attendance/leave using optimized processing"""
    global STREAM, FACE_SMOOTHER
    
    try:
        # Initialize camera stream if not already running
        if STREAM is None:
            STREAM = CameraStream(0)
            STREAM.start()
        
        recognized_ids = set()
        frame_count = 0
        last_recognition_time = {}  # Track last recognition time for each face
        
        # Create a named window with specific properties
        window_name = "Face Recognition - Press 'w' to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, STREAM.actual_width, STREAM.actual_height)
        
        while True:
            frame = STREAM.read()
            if frame is None:
                continue
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
            
            # Process frame for detection
            display_frame, face_locations, face_encodings = process_frame_for_detection(frame)
            if display_frame is None:
                continue
            
            # Draw status bar
            status_bar = display_frame[-STATUS_BAR_HEIGHT:].copy()
            cv2.rectangle(status_bar, (0, 0), (status_bar.shape[1], STATUS_BAR_HEIGHT), 
                         STATUS_BAR_COLOR, -1)
            cv2.addWeighted(status_bar, 0.7, display_frame[-STATUS_BAR_HEIGHT:], 0.3, 0, 
                           display_frame[-STATUS_BAR_HEIGHT:])
            
            # Add status information
            mode_text = "Attendance Mode" if mode == 'attend' else "Leave Mode"
            cv2.putText(display_frame, mode_text, 
                       (20, display_frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            
            cv2.putText(display_frame, "Press 'w' to exit", 
                       (display_frame.shape[1] - 200, display_frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            
            # Add face count
            cv2.putText(display_frame, f"Faces detected: {len(face_locations)}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
            
            current_time = time.time()
            
            for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # Check if face is too small
                face_height = bottom - top
                if face_height < FACE_DETECTION_MIN_SIZE:
                    continue
                    
                # Check cache first
                face_key = tuple(face_encoding)
                if face_key in FACE_ENCODING_CACHE:
                    emp_id, name, confidence = FACE_ENCODING_CACHE[face_key]
                else:
                    # Process face recognition with increased tolerance
                    matches = face_recognition.compare_faces(
                        KNOWN_ENCODINGS, 
                        face_encoding,
                        tolerance=FACE_RECOGNITION_TOLERANCE
                    )
                    face_distances = face_recognition.face_distance(KNOWN_ENCODINGS, face_encoding)
                    
                    if len(face_distances) == 0:
                        continue
                    
                    best_match_index = np.argmin(face_distances)
                    match_distance = face_distances[best_match_index]
                    confidence = 1 - match_distance
                    
                    # Apply distance-based confidence adjustment
                    if face_height < 50:  # For smaller faces (longer range)
                        confidence *= 1.2  # Boost confidence for long-range detections
                    
                    if matches[best_match_index]:
                        emp_id = int(KNOWN_IDS[best_match_index])
                        name = KNOWN_NAMES[best_match_index]
                        # Cache the result
                        FACE_ENCODING_CACHE[face_key] = (emp_id, name, confidence)
                    else:
                        emp_id = None
                        name = None
                
                # Apply smoothing with face location
                if emp_id is not None:
                    smoothed_result, smoothed_confidence = FACE_SMOOTHER.update(
                        i, 
                        (emp_id, name), 
                        confidence,
                        ((top + bottom) // 2, (left + right) // 2)  # Face center point
                    )
                    if smoothed_result is not None:
                        emp_id, name = smoothed_result
                        confidence = smoothed_confidence
                    else:
                        emp_id = None
                        name = None
                else:
                    FACE_SMOOTHER.reset(i)
                
                if emp_id is not None and emp_id not in EMPLOYEE_MAP:
                    flash(f"Unknown employee with ID: {emp_id}. Attendance not recorded.", "error")
                    continue
                
                # Check if enough time has passed since last recognition
                if emp_id in last_recognition_time:
                    time_since_last = current_time - last_recognition_time[emp_id]
                    if time_since_last < 2.0:  # Minimum 2 seconds between recognitions
                        continue
                
                if emp_id is not None and emp_id not in recognized_ids:
                    if mode == 'attend':
                        register_attendance(emp_id, name)
                    elif mode == 'leave':
                        register_leave(emp_id, name)
                    recognized_ids.add(emp_id)
                    last_recognition_time[emp_id] = current_time
                
                # Update visualization
                color = (0, int(255 * confidence), 0)
                if emp_id is not None:
                    # Draw rectangle with gradient color based on match confidence
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    label = f"{name} (ID: {emp_id})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(display_frame, 
                                (left, top - label_size[1] - 10),
                                (left + label_size[0], top),
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(display_frame, label,
                              (left, top - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
                    
                    # Add confidence score
                    confidence_text = f"Confidence: {confidence:.2%}"
                    cv2.putText(display_frame, confidence_text,
                              (left, bottom + 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # Draw red rectangle for unknown faces
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(display_frame, "Unknown",
                              (left, top - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('w'):
                break
                
    finally:
        cv2.destroyAllWindows()
        write_batched_attendance()

    if not recognized_ids:
        flash("No known faces recognized. No attendance recorded.", "error")

def calculate_attendance_metrics():
    """Get raw attendance data for display"""
    try:
        attendance = pd.read_excel(ATTENDANCE_FILE, engine='openpyxl') if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame()
        employee_data = pd.read_excel(EMPLOYEE_FILE, engine='openpyxl') if os.path.exists(EMPLOYEE_FILE) else pd.DataFrame()
        
        # Merge attendance data with employee data to get department information
        if not attendance.empty and not employee_data.empty:
            attendance = pd.merge(
                attendance,
                employee_data[['ID', 'Department']],
                on='ID',
                how='left'
            )
        
        # Convert DataFrame to list of dictionaries
        attendance_records = attendance.to_dict('records')
        return attendance_records
    except Exception as e:
        print(f"Error getting attendance data: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['admin'] = username
        return redirect(url_for('dashboard'))
    else:
        flash("Incorrect username or password!", "error")
        return redirect(url_for('home', error='Incorrect username or password!'))

@app.route('/dashboard')
def dashboard():
    if 'admin' in session:
        try:
            metrics = calculate_attendance_metrics()
            # Get unique departments for filter
            departments = []
            try:
                df = pd.read_excel(EMPLOYEE_FILE, engine='openpyxl')
                departments = sorted(df['Department'].unique().tolist())
            except Exception as e:
                print(f"Error reading employee file: {e}")
                # If file is corrupted, reinitialize it
                init_employee_file()
            return render_template('dashboard.html', metrics=metrics, departments=departments)
        except Exception as e:
            print(f"Error in dashboard: {e}")
            flash("Error loading dashboard data. Please try again.", "error")
            return redirect(url_for('home'))
    else:
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect(url_for('home'))

@app.route('/add', methods=['POST'])
def add():
    emp_id = int(request.form['id'])
    name = request.form['name']
    department = request.form['department']
    init_employee_file()
    if add_employee(emp_id, name, department, None):
        capture_employee_photos(emp_id, name)
        flash("Student added and photos captured successfully!", "success")
    else:
        flash("Student already exists!", "error")
    return redirect('/')

@app.route('/face_attend')
def face_attend():
    recognize_face_and_register(mode='attend')
    return redirect(url_for('home'))

@app.route('/face_leave')
def face_leave():
    recognize_face_and_register(mode='leave')
    return redirect(url_for('home'))

def delete_employee_data(emp_id):
    """Delete employee data from all relevant files and caches"""
    try:
        # Delete from employee Excel file
        if os.path.exists(EMPLOYEE_FILE):
            df = pd.read_excel(EMPLOYEE_FILE, engine='openpyxl')
            df = df[df['ID'] != emp_id]
            df.to_excel(EMPLOYEE_FILE, index=False, engine='openpyxl')
        
        # Delete from attendance Excel file
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_excel(ATTENDANCE_FILE, engine='openpyxl')
            df = df[df['ID'] != emp_id]
            df.to_excel(ATTENDANCE_FILE, index=False, engine='openpyxl')
        
        # Delete from face database
        global KNOWN_ENCODINGS, KNOWN_IDS, KNOWN_NAMES
        indices_to_keep = [i for i, id in enumerate(KNOWN_IDS) if id != str(emp_id)]
        KNOWN_ENCODINGS = [KNOWN_ENCODINGS[i] for i in indices_to_keep]
        KNOWN_IDS = [KNOWN_IDS[i] for i in indices_to_keep]
        KNOWN_NAMES = [KNOWN_NAMES[i] for i in indices_to_keep]
        
        # Save updated face database
        with open(FACE_DB_FILE, 'wb') as f:
            pickle.dump((KNOWN_ENCODINGS, KNOWN_IDS, KNOWN_NAMES), f)
        
        # Delete from employee map
        if emp_id in EMPLOYEE_MAP:
            del EMPLOYEE_MAP[emp_id]
        
        # Delete face images
        for folder in os.listdir('dataset'):
            if folder.startswith(f"{emp_id}_"):
                folder_path = os.path.join('dataset', folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        os.remove(os.path.join(folder_path, file))
                    os.rmdir(folder_path)
        
        return True
    except Exception as e:
        print(f"Error deleting employee data: {e}")
        return False

@app.route('/delete_employee', methods=['POST'])
def delete_employee():
    """Handle employee deletion request"""
    if 'admin' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized access'})
    
    try:
        emp_id = int(request.form['emp_id'])
        
        # Verify employee exists
        if not os.path.exists(EMPLOYEE_FILE):
            return jsonify({'success': False, 'message': 'Employee database not found'})
        
        df = pd.read_excel(EMPLOYEE_FILE, engine='openpyxl')
        if emp_id not in df['ID'].values:
            return jsonify({'success': False, 'message': 'Employee not found'})
        
        # Delete employee data
        if delete_employee_data(emp_id):
            return jsonify({'success': True, 'message': 'Employee deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Error deleting employee data'})
            
    except Exception as e:
        print(f"Error in delete_employee: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while deleting the employee'})

# Add cleanup on application exit
@app.teardown_appcontext
def cleanup(exception=None):
    global STREAM
    if STREAM is not None:
        STREAM.stop()
        STREAM = None

if __name__ == '__main__':
    app.run(debug=True)
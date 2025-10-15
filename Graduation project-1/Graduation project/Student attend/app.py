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
import dlib

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Performance and Quality Tuning Parameters
# Camera resolution settings (will fall back to lower resolution if not supported)
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
FALLBACK_WIDTH = 1280
FALLBACK_HEIGHT = 720

# Frame processing settings
FRAME_SKIP = 4  # Process every Nth frame
DETECTION_RESIZE_SCALE = 0.75  # Resize frame before detection (0.75 = 75% of original size)
UPSAMPLE = 0  # Face detection upsampling (0 for speed, 1 for accuracy)
FACE_DETECTION_MODEL = 'cnn'  # 'hog' for CPU, 'cnn' for GPU (requires CUDA-enabled dlib)

# Range detection settings
KNOWN_FACE_HEIGHT_M = 0.16  # Average face height in meters
FOCAL_LENGTH_PX = 1200  # Increased focal length for better range
MIN_RANGE_M = 3.0  # Minimum recognition range in meters
MAX_RANGE_M = 8.0  # Increased maximum recognition range in meters
FACE_DETECTION_CONFIDENCE = 0.3  # Lower confidence threshold for better range detection

# Initialize dlib face detector
FACE_DETECTOR = dlib.get_frontal_face_detector()

# Camera configuration
CAMERA_CONFIG = {
    'ip': os.getenv('CAMERA_IP', '192.168.1.4'),
    'username': os.getenv('CAMERA_USER', 'martin'),
    'password': os.getenv('CAMERA_PASS', 'martin'),
    'port': int(os.getenv('CAMERA_PORT', '554')),
    'stream': os.getenv('CAMERA_STREAM', 'stream1'),
    'rtsp_transport': 'tcp',
    'buffer_size': 2048 * 1024,  # Balanced buffer size
    'timeout': 10000,  # Balanced timeout
    'reconnect_delay': 1,
    'max_reconnect_attempts': 5,
    'cv2_config': {
        cv2.CAP_PROP_BUFFERSIZE: 5,  # Balanced buffer size
        cv2.CAP_PROP_FPS: 30,
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'H264'),
        cv2.CAP_PROP_FRAME_WIDTH: CAMERA_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT: CAMERA_HEIGHT,
        cv2.CAP_PROP_AUTOFOCUS: 1,
        cv2.CAP_PROP_AUTO_EXPOSURE: 1,
        cv2.CAP_PROP_GAIN: 1.1,  # Slightly increased gain
        cv2.CAP_PROP_BRIGHTNESS: 1.05,  # Slightly increased brightness
        cv2.CAP_PROP_CONTRAST: 1.05  # Slightly increased contrast
    }
}

# UI settings
STATUS_BAR_HEIGHT = 40
STATUS_BAR_COLOR = (50, 50, 50)  # Dark gray
TEXT_COLOR = (255, 255, 255)  # White

# File paths
EMPLOYEE_FILE = 'students.xlsx'
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
        """Construct RTSP URL from camera configuration"""
        return f"rtsp://{CAMERA_CONFIG['username']}:{CAMERA_CONFIG['password']}@{CAMERA_CONFIG['ip']}:{CAMERA_CONFIG['port']}/{CAMERA_CONFIG['stream']}"
        
    def _connect(self):
        """Attempt to connect to the camera with dynamic resolution"""
        try:
            # Construct RTSP URL
            rtsp_url = self._get_rtsp_url()
            
            # Set up RTSP transport
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f"rtsp_transport;{CAMERA_CONFIG['rtsp_transport']}"
            
            # Open video capture with RTSP URL
            self.stream = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            
            if not self.stream.isOpened():
                raise RuntimeError("Failed to open camera stream")
            
            # Apply CV2 configuration
            for prop, value in CAMERA_CONFIG['cv2_config'].items():
                self.stream.set(prop, value)
            
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
            if self.reconnect_attempts >= CAMERA_CONFIG['max_reconnect_attempts']:
                raise RuntimeError(f"Failed to connect to camera after {CAMERA_CONFIG['max_reconnect_attempts']} attempts: {str(e)}")
            time.sleep(CAMERA_CONFIG['reconnect_delay'])
            return False
        
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
                        time.sleep(CAMERA_CONFIG['reconnect_delay'])  # Wait before retry
                        continue
                    continue
                    
                ret, frame = self.stream.retrieve()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame.copy()  # Store a copy of the frame
            except Exception as e:
                print(f"Error in camera stream: {e}")
                time.sleep(CAMERA_CONFIG['reconnect_delay'])  # Wait before retry
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
            df = pd.read_excel(EMPLOYEE_FILE)
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

def load_employee_data():
    return pd.read_excel(EMPLOYEE_FILE)

def load_attendance_data():
    return pd.read_excel(ATTENDANCE_FILE)


def init_employee_file():
    if not os.path.exists(EMPLOYEE_FILE):
        df = pd.DataFrame(columns=['ID', 'Name', 'Department'])
        df.to_excel(EMPLOYEE_FILE, index=False)


def init_attendance_file():
    """Initialize attendance Excel file with proper structure"""
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time'])
        df.to_excel(ATTENDANCE_FILE, index=False)
    else:
        # Verify file structure
        try:
            df = pd.read_excel(ATTENDANCE_FILE)
            if 'ID' not in df.columns:
                # If ID column is missing, create new file
                df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time'])
                df.to_excel(ATTENDANCE_FILE, index=False)
        except:
            # If file is corrupted, create new one
            df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time'])
            df.to_excel(ATTENDANCE_FILE, index=False)


def add_employee(emp_id, name, department):
    """Add new student and update the cached student map"""
    global EMPLOYEE_MAP
    
    if emp_id in EMPLOYEE_MAP:
        return False
        
    # Add to Excel file
    if os.path.exists(EMPLOYEE_FILE):
        df = pd.read_excel(EMPLOYEE_FILE)
    else:
        df = pd.DataFrame(columns=['ID', 'Name', 'Department'])
    
    new_data = {'ID': emp_id, 'Name': name, 'Department': department}
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_excel(EMPLOYEE_FILE, index=False)
    
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
                
            cv2.imshow('Capturing Photos - Press q to quit', frame)
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
        'Attendance Time': time_now
    })
    flash(f"Attendance recorded for {name} at {time_now}.", "success")


def write_batched_attendance():
    """Write all batched attendance records to the Excel file"""
    if not NEW_ATTENDANCE_ROWS:
        return
    
    try:
        # Read existing attendance records
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_excel(ATTENDANCE_FILE)
        else:
            df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Attendance Time'])
        
        # Append new records
        new_df = pd.DataFrame(NEW_ATTENDANCE_ROWS)
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Write back to file
        df.to_excel(ATTENDANCE_FILE, index=False)
        print(f"Wrote {len(NEW_ATTENDANCE_ROWS)} attendance records to {ATTENDANCE_FILE}")
        
        # Clear the batch queue
        NEW_ATTENDANCE_ROWS.clear()
    except Exception as e:
        print(f"Error writing batched attendance: {e}")
        flash("Error saving attendance records.", "error")


def estimate_distance(box_height_px):
    """Estimate distance to face using pinhole camera model with range compensation"""
    if box_height_px == 0:
        return float('inf')
    # Apply range compensation factor based on distance
    base_distance = (KNOWN_FACE_HEIGHT_M * FOCAL_LENGTH_PX) / box_height_px
    # Compensate for perspective distortion at longer ranges
    if base_distance > 5.0:
        compensation_factor = 1.0 + (base_distance - 5.0) * 0.1
        return base_distance * compensation_factor
    return base_distance

def process_frame_for_detection(frame):
    """Process frame for face detection with configurable parameters"""
    if frame is None:
        return None, None, None
    
    # Calculate new dimensions based on resize scale
    height, width = frame.shape[:2]
    new_width = int(width * DETECTION_RESIZE_SCALE)
    new_height = int(height * DETECTION_RESIZE_SCALE)
    
    # Resize frame for detection
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Use dlib face detector for better range performance
    dlib_faces = FACE_DETECTOR(rgb_frame, 1)  # 1 is the upsampling factor
    
    # Convert dlib rectangles to face_recognition format
    face_locations = []
    for face in dlib_faces:
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        left = face.left()
        face_locations.append((top, right, bottom, left))
    
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

def recognize_face_and_register():
    """Recognize faces and register attendance using cached data"""
    global STREAM
    
    try:
        # Initialize camera stream if not already running
        if STREAM is None:
            STREAM = CameraStream(0)
            STREAM.start()
        
        recognized_ids = set()
        frame_count = 0
        
        # Create a named window with specific properties
        window_name = "Face Recognition - Press 'w' to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, STREAM.actual_width, STREAM.actual_height)
        
        while True:
            frame = STREAM.read()
            if frame is None:
                continue
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:  # Use configurable frame skip
                continue
            
            # Process frame for detection
            display_frame, face_locations, face_encodings = process_frame_for_detection(frame)
            if display_frame is None:
                continue
            
            # Draw status bar (lightweight version)
            status_bar = display_frame[-STATUS_BAR_HEIGHT:].copy()
            cv2.rectangle(status_bar, (0, 0), (status_bar.shape[1], STATUS_BAR_HEIGHT), 
                         STATUS_BAR_COLOR, -1)
            cv2.addWeighted(status_bar, 0.7, display_frame[-STATUS_BAR_HEIGHT:], 0.3, 0, 
                           display_frame[-STATUS_BAR_HEIGHT:])
            
            # Add status information
            cv2.putText(display_frame, "Attendance Mode", 
                       (20, display_frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            
            cv2.putText(display_frame, "Press 'w' to exit", 
                       (display_frame.shape[1] - 200, display_frame.shape[0] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            
            # Add face count
            cv2.putText(display_frame, f"Faces detected: {len(face_locations)}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(KNOWN_ENCODINGS, face_encoding)
                face_distances = face_recognition.face_distance(KNOWN_ENCODINGS, face_encoding)
                
                if len(face_distances) == 0:
                    continue
                
                best_match_index = np.argmin(face_distances)
                match_distance = face_distances[best_match_index]
                
                if matches[best_match_index]:
                    emp_id = int(KNOWN_IDS[best_match_index])
                    name = KNOWN_NAMES[best_match_index]
                    
                    if emp_id not in EMPLOYEE_MAP:
                        flash(f"Unknown student with ID: {emp_id}. Attendance not recorded.", "error")
                        continue
                    
                    if emp_id not in recognized_ids:
                        register_attendance(emp_id, name)
                        recognized_ids.add(emp_id)
                    
                    # Draw rectangle with gradient color based on match confidence
                    confidence = 1 - match_distance
                    color = (0, int(255 * confidence), 0)
                    
                    # Draw rectangle and label
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
    """Calculate attendance metrics using cached student data"""
    # Ensure attendance file exists with correct structure
    init_attendance_file()
    
    try:
        attendance = pd.read_excel(ATTENDANCE_FILE)
        student_metrics = []
        
        for emp_id, name in EMPLOYEE_MAP.items():
            try:
                # Get department from Excel file (only needed for metrics)
                df = pd.read_excel(EMPLOYEE_FILE)
                emp_data = df[df['ID'] == emp_id].iloc[0]
                department = emp_data['Department']
                
                emp_attendance = attendance[attendance['ID'] == emp_id]
                days_attended = len(emp_attendance[emp_attendance['Attendance Time'].notnull()])
                days_absent = len(emp_attendance[emp_attendance['Attendance Time'].isnull()])
                
                if days_absent > 5:
                    color_class = 'red'
                elif days_absent == 3:
                    color_class = 'yellow'
                else:
                    color_class = 'green'
                
                student_metrics.append({
                    'ID': emp_id,
                    'Name': name,
                    'Department': department,
                    'Days Attended': days_attended,
                    'Days Absent': days_absent,
                    'Color Class': color_class
                })
            except:
                # Skip this student if there's an error
                continue
        
        return student_metrics
    except:
        # Return empty list if there's an error
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
        metrics = calculate_attendance_metrics()
        return render_template('dashboard.html' , metrics=metrics)
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
    if add_employee(emp_id, name, department):
        capture_employee_photos(emp_id, name)
        flash("Student added and photos captured successfully!", "success")
    else:
        flash("Student already exists!", "error")
    return redirect('dashboard')

@app.route('/face_attend')
def face_attend():
    recognize_face_and_register()
    return redirect(url_for('dashboard'))

# Add cleanup on application exit
@app.teardown_appcontext
def cleanup(exception=None):
    global STREAM
    if STREAM is not None:
        STREAM.stop()
        STREAM = None

if __name__ == '__main__':
    app.run(debug=True)
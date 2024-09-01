import numpy as np
import cv2 as cv

def initialize_camera(camera_index=0):
    """Initializes the camera capture."""
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError('Could not capture the camera!')
    return cap

def read_frame(cap):
    """Reads a frame from the camera."""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError('Could not receive frame!')
    return frame

def initialize_tracking_window(x=100, y=100, w=100, h=100):
    """Defines the initial tracking window."""
    return (x, y, w, h)

def set_roi_hist(frame, track_window):
    """Sets the Region of Interest (ROI) for tracking."""
    x, y, w, h = track_window
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])
    return cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

def load_cascades():
    """Loads the pre-trained classifiers for face and eyes detection."""
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError('Could not load the Haar cascade files!')
    return face_cascade, eye_cascade

def detect_faces_and_eyes(frame, face_cascade, eye_cascade):
    """Detects faces and eyes in the frame."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (125, 255, 125), 2)
    return frame

def apply_tracking_and_detection(cap, face_cascade, eye_cascade):
    """Main loop that applies tracking and detection algorithms."""
    track_window = initialize_tracking_window()
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    
    # Get the first frame and set ROI histogram
    frame = read_frame(cap)
    roi_hist = set_roi_hist(frame, track_window)

    while True:
        frame = read_frame(cap)

        # Convert image to HSV and create masks for red objects
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask_1 = cv.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask_2 = cv.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = mask_1 + mask_2
        result = cv.bitwise_and(frame, frame, mask=mask)

        # Apply MeanShift
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        _, track_window = cv.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        result = cv.rectangle(result, (x, y), (x+w, y+h), 255, 2)

        # Apply CamShift
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        result = cv.polylines(result, [pts], True, 255, 2)

        # Apply Canny edge detection
        edge = cv.Canny(result, 100, 200, apertureSize=5, L2gradient=True)

        # Detect faces and eyes
        frame = detect_faces_and_eyes(frame, face_cascade, eye_cascade)

        # Draw contours and other shapes
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if 50 <= w <= 100 and 50 <= h <= 100:
                cv.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                center_x = x + w // 2
                center_y = y + h // 2
                cv.circle(result, (center_x, center_y), 5, (0, 255, 255), -1)
                rect = cv.minAreaRect(contour)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(result, [box], 0, (0, 0, 255), 2)

        # Display the frames
        font = cv.FONT_HERSHEY_COMPLEX
        cv.putText(frame, 'Face and Eyes', (0, 100), font, 1, (255, 255, 255), 1)
        cv.putText(result, 'Tracking', (0, 100), font, 1, (255, 255, 255), 1)
        cv.putText(edge, 'Canny Edge', (0, 100), font, 1, (255, 255, 255), 1)
        cv.imshow('contours', frame)
        cv.imshow('result', result)
        cv.imshow('edge', edge)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    """Main function to run the object tracking and face/eye detection."""
    try:
        cap = initialize_camera(0)
        face_cascade, eye_cascade = load_cascades()
        apply_tracking_and_detection(cap, face_cascade, eye_cascade)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()

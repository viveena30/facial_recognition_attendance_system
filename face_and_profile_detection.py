
import cv2

# Load video
video = cv2.VideoCapture(0)

# Load Haar cascades
face_detect_frontal = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
face_detect_profile = cv2.CascadeClassifier('data/haarcascade_profileface.xml')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using frontal face cascade
    faces_frontal = face_detect_frontal.detectMultiScale(gray, 1.3, 5)
    # Detect faces using profile face cascade
    faces_profile = face_detect_profile.detectMultiScale(gray, 1.3, 5)
    
    # Draw rectangles around detected faces (frontal)
    for (x, y, w, h) in faces_frontal:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    
    # Draw rectangles around detected faces (profile)
    for (x, y, w, h) in faces_profile:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 50, 50), 1)
    
    # Show the frame with detected faces
    cv2.imshow('frame', frame)
    
    # Wait for 1 ms and check if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and destroy windows
video.release()
cv2.destroyAllWindows()

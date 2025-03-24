import cv2
from fer import FER

# Load FER pre-trained model
detector = FER()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Ensure face ROI does not exceed frame boundaries
        x_end, y_end = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
        face = frame[y:y_end, x:x_end]

        # Detect emotions using FER model
        emotion_results = detector.detect_emotions(face)

        if emotion_results:
            # Get the most likely emotion
            emotions = emotion_results[0]["emotions"]
            emotion = max(emotions, key=emotions.get)

            # Draw rectangle around face and display emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show frame
    cv2.imshow("Mood Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

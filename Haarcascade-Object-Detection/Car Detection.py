import cv2
import time

# Load the car cascade classifier
car_classifier_path = r'C:\Users\Prachi\Documents\VS Code Files\DEEP LEARNING\OPEN CV\Haarcascades\Haarcascades\haarcascade_car.xml'
car_classifier = cv2.CascadeClassifier(car_classifier_path)

# Check if the classifier is loaded correctly
if car_classifier.empty():
    print(f"Error: Could not load the car classifier at {car_classifier_path}. Make sure the path is correct.")
    exit()


# Video path
#video_path = r'C:\Users\Prachi\Documents\VS Code Files\DEEP LEARNING\OPEN CV\Haarcascades\1900-151662242_small.mp4'
video_path = r'C:\Users\Prachi\Documents\VS Code Files\DEEP LEARNING\OPEN CV\Haarcascades\127769236-c6c65f7f-1450-4d14-b150-42b0e5077dc9.mp4'


# Load the video
cap = cv2.VideoCapture(video_path)

# Check if the video file is loaded correctly
if not cap.isOpened():
    print(f"Error: Could not open the video at {video_path}. Make sure the file path is correct.")
    exit()

print("Video opened successfully. Starting car detection...")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame or video has ended.")
        break

    # Convert the frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the grayscale image
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw rectangles around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the frame with the detected cars
    cv2.imshow('Cars Detection', frame)

    # Exit the loop when the Enter key is pressed
    if cv2.waitKey(1) == 13:  # 13 is the Enter key
        print("Exiting...")
        break

    # Optional: Add a small delay between frames (to make the processing smoother)
    #time.sleep(0.05)

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()






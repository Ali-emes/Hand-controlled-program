import cv2

# Open the default webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    cv2.imshow("Webcam Feed - Press Q to quit", frame)

    # Break loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
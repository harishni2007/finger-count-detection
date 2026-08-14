import cv2

print("Starting camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera cannot be opened.")
    exit()

print("Camera opened successfully.")

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Cannot read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "CAMERA WORKING",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Camera Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
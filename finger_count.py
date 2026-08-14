import cv2
import mediapipe as mp
import math
from collections import deque, Counter


# --------------------------------------------------
# MediaPipe Setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)


landmarker = HandLandmarker.create_from_options(options)


# --------------------------------------------------
# Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


timestamp = 0


# --------------------------------------------------
# Stable Count History
# --------------------------------------------------

count_history = deque(maxlen=8)


# --------------------------------------------------
# Calculate Distance
# --------------------------------------------------

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


# --------------------------------------------------
# Calculate Angle
# --------------------------------------------------

def angle(a, b, c):

    ba = [
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ]

    bc = [
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ]

    dot_product = (
        ba[0] * bc[0] +
        ba[1] * bc[1] +
        ba[2] * bc[2]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2 +
        ba[2] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2 +
        bc[2] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    value = dot_product / (magnitude_ba * magnitude_bc)

    value = max(-1, min(1, value))

    return math.degrees(math.acos(value))


# --------------------------------------------------
# Check Normal Finger
# --------------------------------------------------

def is_finger_extended(hand, mcp, pip, dip, tip):

    finger_angle = angle(
        hand[mcp],
        hand[pip],
        hand[dip]
    )

    tip_distance = distance(
        hand[tip],
        hand[0]
    )

    pip_distance = distance(
        hand[pip],
        hand[0]
    )

    # Finger must be reasonably straight
    # and fingertip must be farther from wrist
    # than the PIP joint.

    if finger_angle > 155 and tip_distance > pip_distance * 1.08:
        return True

    return False


# --------------------------------------------------
# Check Thumb
# --------------------------------------------------

def is_thumb_extended(hand):

    thumb_angle = angle(
        hand[2],
        hand[3],
        hand[4]
    )

    tip_distance = distance(
        hand[4],
        hand[0]
    )

    ip_distance = distance(
        hand[3],
        hand[0]
    )

    if thumb_angle > 145 and tip_distance > ip_distance * 1.10:
        return True

    return False


# --------------------------------------------------
# Count Fingers
# --------------------------------------------------

def count_fingers(hand):

    count = 0

    # -----------------------------
    # Thumb
    # -----------------------------

    if is_thumb_extended(hand):
        count += 1


    # -----------------------------
    # Index Finger
    # -----------------------------

    if is_finger_extended(
        hand,
        5,
        6,
        7,
        8
    ):
        count += 1


    # -----------------------------
    # Middle Finger
    # -----------------------------

    if is_finger_extended(
        hand,
        9,
        10,
        11,
        12
    ):
        count += 1


    # -----------------------------
    # Ring Finger
    # -----------------------------

    if is_finger_extended(
        hand,
        13,
        14,
        15,
        16
    ):
        count += 1


    # -----------------------------
    # Little Finger
    # -----------------------------

    if is_finger_extended(
        hand,
        17,
        18,
        19,
        20
    ):
        count += 1


    return count


# --------------------------------------------------
# Draw Hand
# --------------------------------------------------

def draw_hand(frame, hand):

    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]


    # Draw connections

    for start, end in connections:

        x1 = int(hand[start].x * frame.shape[1])
        y1 = int(hand[start].y * frame.shape[0])

        x2 = int(hand[end].x * frame.shape[1])
        y2 = int(hand[end].y * frame.shape[0])

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


    # Draw landmarks

    for point in hand:

        x = int(point.x * frame.shape[1])
        y = int(point.y * frame.shape[0])

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )


# --------------------------------------------------
# Main Loop
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # Mirror camera

    frame = cv2.flip(frame, 1)


    # Convert BGR to RGB

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe Image

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # Timestamp

    timestamp += 33


    # Detect hands

    result = landmarker.detect_for_video(
        image,
        timestamp
    )


    current_total = 0


    # --------------------------------------------------
    # Process Detected Hands
    # --------------------------------------------------

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            # Count fingers

            finger_count = count_fingers(hand)

            current_total += finger_count


            # Draw hand

            draw_hand(
                frame,
                hand
            )


            # ------------------------------------------
            # Individual Hand Count
            # ------------------------------------------

            x = int(hand[0].x * frame.shape[1])
            y = int(hand[0].y * frame.shape[0])


            cv2.putText(
                frame,
                "Fingers: " + str(finger_count),
                (x - 60, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


    # --------------------------------------------------
    # Stabilize Count
    # --------------------------------------------------

    count_history.append(current_total)


    if len(count_history) >= 5:

        most_common = Counter(
            count_history
        ).most_common(1)[0]

        stable_count = most_common[0]

    else:

        stable_count = current_total


    # --------------------------------------------------
    # Display Total Count
    # --------------------------------------------------

    cv2.putText(
        frame,
        "Finger Count: " + str(stable_count),
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3
    )


    # --------------------------------------------------
    # Display Status
    # --------------------------------------------------

    if stable_count == 0:

        status = "No Fingers"

    elif stable_count == 1:

        status = "One Finger"

    elif stable_count == 2:

        status = "Two Fingers"

    elif stable_count == 3:

        status = "Three Fingers"

    elif stable_count == 4:

        status = "Four Fingers"

    elif stable_count == 5:

        status = "Five Fingers"

    elif stable_count == 6:

        status = "Six Fingers"

    elif stable_count == 7:

        status = "Seven Fingers"

    elif stable_count == 8:

        status = "Eight Fingers"

    elif stable_count == 9:

        status = "Nine Fingers"

    elif stable_count == 10:

        status = "Ten Fingers"

    else:

        status = str(stable_count) + " Fingers"


    cv2.putText(
        frame,
        status,
        (40, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------
    # Show Camera
    # --------------------------------------------------

    cv2.imshow(
        "Finger Count Detection",
        frame
    )


    # --------------------------------------------------
    # Press Q to Exit
    # --------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Release Resources
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

landmarker.close()
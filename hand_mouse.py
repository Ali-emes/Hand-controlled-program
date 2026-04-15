import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ── Safety: pyautogui won't throw an exception if the cursor hits a screen edge
pyautogui.FAILSAFE = False

# ── Screen resolution (e.g. 1920 x 1080)
SCREEN_W, SCREEN_H = pyautogui.size()

# ── Pinch detection settings
PINCH_THRESHOLD = 40   # pixels in camera space — tune this up/down to taste
CLICK_COOLDOWN  = 0.4  # seconds between allowed clicks
last_click_time = 0.0

# ── MediaPipe setup
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,          # one hand keeps CPU usage low
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ── Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Hand mouse running — press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame so movements feel natural (like a mirror)
    frame = cv2.flip(frame, 1)
    cam_h, cam_w = frame.shape[:2]

    # MediaPipe expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # ── Draw skeleton on the frame
        mp_draw.draw_landmarks(
            frame, hand,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        lm = hand.landmark  # shorthand

        # ── Landmark 8 = index finger tip
        # lm[8].x and lm[8].y are normalised [0.0 – 1.0] relative to frame size.
        # Multiply by frame dimensions to get pixel coords in camera space.
        ix = int(lm[8].x * cam_w)
        iy = int(lm[8].y * cam_h)

        # ── Screen mapping (simple proportion):
        #    screen_x / SCREEN_W  =  camera_x / cam_w
        #    => screen_x = camera_x * (SCREEN_W / cam_w)
        sx = int(lm[8].x * SCREEN_W)
        sy = int(lm[8].y * SCREEN_H)
        pyautogui.moveTo(sx, sy, duration=0)   # duration=0 → instant, no lag

        # ── Landmark 4 = thumb tip
        tx = int(lm[4].x * cam_w)
        ty = int(lm[4].y * cam_h)

        # ── Euclidean distance between index tip and thumb tip (in camera pixels):
        #    d = sqrt( (x2-x1)^2 + (y2-y1)^2 )
        distance = math.hypot(tx - ix, ty - iy)

        # ── Pinch detection with cooldown
        now = time.time()
        pinching = distance < PINCH_THRESHOLD
        if pinching and (now - last_click_time) > CLICK_COOLDOWN:
            pyautogui.click()
            last_click_time = now

        # ── Visual feedback on frame
        color = (0, 0, 255) if pinching else (0, 255, 0)  # red = pinch, green = open
        cv2.circle(frame, (ix, iy), 10, color, -1)         # index tip
        cv2.circle(frame, (tx, ty), 10, color, -1)         # thumb tip
        cv2.line(frame, (ix, iy), (tx, ty), color, 2)      # line between them

        # Distance label
        mid = ((ix + tx) // 2, (iy + ty) // 2)
        cv2.putText(frame, f"d={int(distance)}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Pinch indicator
        if pinching:
            cv2.putText(frame, "CLICK!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    cv2.imshow("Hand Mouse — Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Closed.")

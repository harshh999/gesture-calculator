import cv2
import mediapipe as mp
import numpy as np

# ------------- Config -------------
START_X = 50          # left padding
START_Y = 150         # top padding
BTN_W, BTN_H = 100, 100
GAP = 12              # space between buttons
PINCH_THRESHOLD = 35  # pixels
DEBOUNCE_FRAMES = 10
PRESS_FEEDBACK_FRAMES = 6

BUTTON_GRID = [
    ['7', '8', '9', '/'],
    ['4', '5', '6', '*'],
    ['1', '2', '3', '-'],
    ['0', '.', '=', '+'],
    ['C']
]

DISPLAY_H = 70
DISPLAY_Y1 = 50
DISPLAY_Y2 = DISPLAY_Y1 + DISPLAY_H
NUM_COLS = 4
DISPLAY_W = START_X + NUM_COLS * (BTN_W + GAP) - GAP

# ------------- MediaPipe setup -------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


# ------------- UI -------------
class Button:
    def __init__(self, pos, text, size=(BTN_W, BTN_H)):
        self.pos = pos
        self.size = size
        self.text = text

    def draw(self, img, pressed=False):
        x, y = self.pos
        w, h = self.size

        # Glassmorphic background
        overlay = img.copy()
        alpha = 0.35 if pressed else 0.2
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Soft purple border
        border_color = (180, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 2)

        # Center text
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(img, self.text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return img

    def hit(self, x, y):
        bx, by = self.pos
        bw, bh = self.size
        return bx < x < bx + bw and by < y < by + bh


# Build the button objects
buttons = []
for i, row in enumerate(BUTTON_GRID):
    for j, val in enumerate(row):
        x = START_X + j * (BTN_W + GAP)
        y = START_Y + i * (BTN_H + GAP)
        buttons.append(Button((x, y), val, (BTN_W, BTN_H)))


def draw_display(img, text):
    overlay = img.copy()
    cv2.rectangle(overlay, (START_X, DISPLAY_Y1), (DISPLAY_W, DISPLAY_Y2), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    cv2.rectangle(img, (START_X, DISPLAY_Y1), (DISPLAY_W, DISPLAY_Y2), (180, 0, 255), 2)
    cv2.putText(img, text[-14:], (START_X + 15, DISPLAY_Y2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)


# ------------- Main loop -------------
equation = ""
click_delay = 0
pressed_button = None
pressed_feedback_frames = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    lm_list = []
    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            for idx, lm in enumerate(hand_lms.landmark):
                lm_list.append([idx, int(lm.x * w), int(lm.y * h)])

    # Display bar
    draw_display(frame, equation)

    # Draw buttons
    for b in buttons:
        is_pressed = (b is pressed_button) and (pressed_feedback_frames > 0)
        b.draw(frame, pressed=is_pressed)

    # Gesture detection
    if lm_list:
        x1, y1 = lm_list[8][1], lm_list[8][2]   # index tip
        x2, y2 = lm_list[4][1], lm_list[4][2]   # thumb tip
        dist = np.hypot(x2 - x1, y2 - y1)

        if dist < PINCH_THRESHOLD and click_delay == 0:
            for b in buttons:
                if b.hit(x1, y1):
                    if b.text == '=':
                        try:
                            equation = str(eval(equation))
                        except Exception:
                            equation = "Error"
                    elif b.text == 'C':
                        equation = ""
                    else:
                        equation += b.text

                    pressed_button = b
                    pressed_feedback_frames = PRESS_FEEDBACK_FRAMES
                    click_delay = DEBOUNCE_FRAMES
                    break

    # Timers
    if click_delay > 0:
        click_delay -= 1
    if pressed_feedback_frames > 0:
        pressed_feedback_frames -= 1
    else:
        pressed_button = None

    cv2.imshow("Gesture Calculator (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

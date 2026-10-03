import os
import time
import cv2
import joblib
import numpy as np
import mediapipe as mp


# ============================================================
# AIRWRITE AI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "airwrite_model.pkl"
)

HAND_MODEL_PATH = os.path.join(
    BASE_DIR, "hand_landmarker.task"
)

# IMPORTANT:
# Your current valid face model is face_landmarker.task
FACE_MODEL_PATH = os.path.join(
    BASE_DIR, "face_landmarker.task"
)


# ============================================================
# SETTINGS
# ============================================================

CAMERA_WIDTH = 900
HEIGHT = 720

PANEL_WIDTH = 380

BRUSH_SIZE = 7
SMOOTHING = 0.70

LETTER_PAUSE = 1.5
MIN_PIXELS = 100

SELECT_HOLD_TIME = 0.7


# ============================================================
# DIGITAL NOTE SETTINGS
# ============================================================

LOCATION_LINE_1 = "Marwadi University"
LOCATION_LINE_2 = "Rajkot, India"

NOTE_BACKGROUND = (245, 245, 235)

NOTE_TEXT = (30, 30, 30)
NOTE_GRAY = (100, 100, 100)

NOTE_BORDER = (80, 180, 80)


# ============================================================
# WRITING COLORS
# OpenCV uses BGR
# ============================================================

RED = (0, 0, 255)

GREEN = (0, 255, 0)

YELLOW = (0, 255, 255)

NAVY_BLUE = (128, 0, 0)


current_color = GREEN
current_color_name = "GREEN"


# ============================================================
# COLOR BUTTON POSITIONS
# ============================================================

BUTTON_Y1 = 430
BUTTON_Y2 = 500

RED_X1 = 20
RED_X2 = 105

GREEN_X1 = 115
GREEN_X2 = 200

YELLOW_X1 = 210
YELLOW_X2 = 285

NAVY_X1 = 295
NAVY_X2 = 365


# ============================================================
# CLEAR BUTTON
# ============================================================

CLEAR_X1 = 20
CLEAR_X2 = 365

CLEAR_Y1 = 530
CLEAR_Y2 = 590


# ============================================================
# VARIABLES
# ============================================================

selection_start_time = None
selection_target = None

recognized_text = ""

previous_point = None

smooth_x = None
smooth_y = None

writing = False

last_draw_time = time.time()

letter_ready = False

timestamp_ms = 0

four_finger_clear_done = False


# ============================================================
# A-Z LABELS
# ============================================================

LETTER_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z"
}


# ============================================================
# CHECK FILES
# ============================================================

print()
print("==============================================")
print("              AIRWRITE AI")
print("==============================================")
print()

print("Checking files...")


if not os.path.exists(MODEL_PATH):
    print("ERROR: A-Z model not found:")
    print(MODEL_PATH)
    exit()


if not os.path.exists(HAND_MODEL_PATH):
    print("ERROR: Hand model not found:")
    print(HAND_MODEL_PATH)
    exit()


if not os.path.exists(FACE_MODEL_PATH):
    print("ERROR: Face Landmarker model not found:")
    print(FACE_MODEL_PATH)
    exit()


print("A-Z model       : FOUND")
print("Hand model      : FOUND")
print("Face model      : FOUND")


# ============================================================
# LOAD A-Z MODEL
# ============================================================

print()
print("Loading A-Z model...")

model = joblib.load(MODEL_PATH)

print("A-Z model loaded!")


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)


# ============================================================
# HAND LANDMARKER
# ============================================================

hand_options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


hand_landmarker = (
    HandLandmarker.create_from_options(
        hand_options
    )
)

print("Hand Landmarker loaded!")


# ============================================================
# FACE LANDMARKER
#
# This replaces the broken face_detector.task.
# face_landmarker.task is already in your project.
# ============================================================

face_options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_faces=1,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


face_landmarker = (
    FaceLandmarker.create_from_options(
        face_options
    )
)

print("Face Landmarker loaded!")


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    HEIGHT
)


if not camera.isOpened():

    print("ERROR: Camera not found!")

    hand_landmarker.close()
    face_landmarker.close()

    exit()


print()
print("Camera started!")

print()
print("----------------------------------------------")
print("HAND CONTROLS")
print("----------------------------------------------")
print("1 finger       = WRITE")
print("2 fingers      = SELECT COLOR")
print("3 fingers      = PAUSE")
print("4 fingers      = CLEAR NOTEPAD")
print("----------------------------------------------")
print("COLORS")
print("RED / GREEN / YELLOW / NAVY BLUE")
print("----------------------------------------------")
print("Q / ESC        = EXIT")
print("----------------------------------------------")
print()


# ============================================================
# CANVAS
# ============================================================

color_canvas = np.zeros(
    (
        HEIGHT,
        CAMERA_WIDTH,
        3
    ),
    dtype=np.uint8
)


gray_canvas = np.zeros(
    (
        HEIGHT,
        CAMERA_WIDTH
    ),
    dtype=np.uint8
)


# ============================================================
# CLEAR DRAWING
# ============================================================

def clear_drawing():

    global previous_point
    global smooth_x
    global smooth_y
    global last_draw_time
    global letter_ready

    color_canvas[:] = 0

    gray_canvas[:] = 0

    previous_point = None

    smooth_x = None
    smooth_y = None

    last_draw_time = time.time()

    letter_ready = False


# ============================================================
# CLEAR DIGITAL NOTEPAD
# ============================================================

def clear_notepad():

    global recognized_text

    recognized_text = ""

    clear_drawing()

    print("DIGITAL NOTEPAD CLEARED")


# ============================================================
# COUNT FINGERS
# ============================================================

def count_fingers(landmarks):

    count = 0

    if landmarks[8].y < landmarks[6].y:
        count += 1

    if landmarks[12].y < landmarks[10].y:
        count += 1

    if landmarks[16].y < landmarks[14].y:
        count += 1

    if landmarks[20].y < landmarks[18].y:
        count += 1

    return count


# ============================================================
# PREPROCESS LETTER
# ============================================================

def preprocess_letter():

    pixel_count = cv2.countNonZero(
        gray_canvas
    )

    if pixel_count < MIN_PIXELS:
        return None


    points = cv2.findNonZero(
        gray_canvas
    )

    if points is None:
        return None


    x, y, w, h = cv2.boundingRect(
        points
    )


    if w < 5 or h < 5:
        return None


    letter = gray_canvas[
        y:y + h,
        x:x + w
    ]


    size = max(
        w,
        h
    )


    square = np.zeros(
        (
            size,
            size
        ),
        dtype=np.uint8
    )


    offset_x = (
        size - w
    ) // 2


    offset_y = (
        size - h
    ) // 2


    square[
        offset_y:offset_y + h,
        offset_x:offset_x + w
    ] = letter


    border = max(
        4,
        size // 8
    )


    square = cv2.copyMakeBorder(

        square,

        border,
        border,
        border,
        border,

        cv2.BORDER_CONSTANT,

        value=0
    )


    resized = cv2.resize(

        square,

        (28, 28),

        interpolation=cv2.INTER_AREA
    )


    pixels = resized.astype(
        np.float32
    )


    return pixels.reshape(
        1,
        784
    )


# ============================================================
# RECOGNIZE LETTER
# ============================================================

def recognize_letter():

    processed = preprocess_letter()

    if processed is None:
        return None, 0


    prediction = model.predict(
        processed
    )[0]


    letter = LETTER_MAP.get(
        int(prediction),
        "?"
    )


    confidence = 0.0


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                processed
            )[0]
        )


        confidence = float(
            np.max(
                probabilities
            )
        )


    return letter, confidence


# ============================================================
# DRAW FACE
#
# Face Landmarker gives landmarks instead of a bounding box.
# We create a box around the detected face landmarks.
# ============================================================

def draw_faces(
    frame,
    face_result
):

    if not face_result.face_landmarks:
        return


    frame_height, frame_width = (
        frame.shape[:2]
    )


    for face_landmarks in face_result.face_landmarks:

        xs = []
        ys = []


        for landmark in face_landmarks:

            x = int(
                landmark.x *
                frame_width
            )

            y = int(
                landmark.y *
                frame_height
            )

            xs.append(x)
            ys.append(y)


        if not xs or not ys:
            continue


        x1 = max(
            0,
            min(xs)
        )

        y1 = max(
            0,
            min(ys)
        )

        x2 = min(
            frame_width - 1,
            max(xs)
        )

        y2 = min(
            frame_height - 1,
            max(ys)
        )


        # Face bounding box
        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            NAVY_BLUE,

            3
        )


        cv2.putText(

            frame,

            "FACE",

            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            NAVY_BLUE,

            2
        )


# ============================================================
# FIND BUTTON
# ============================================================

def get_button(
    panel_x,
    panel_y
):

    if (
        RED_X1 <= panel_x <= RED_X2
        and
        BUTTON_Y1 <= panel_y <= BUTTON_Y2
    ):
        return "RED"


    if (
        GREEN_X1 <= panel_x <= GREEN_X2
        and
        BUTTON_Y1 <= panel_y <= BUTTON_Y2
    ):
        return "GREEN"


    if (
        YELLOW_X1 <= panel_x <= YELLOW_X2
        and
        BUTTON_Y1 <= panel_y <= BUTTON_Y2
    ):
        return "YELLOW"


    if (
        NAVY_X1 <= panel_x <= NAVY_X2
        and
        BUTTON_Y1 <= panel_y <= BUTTON_Y2
    ):
        return "NAVY"


    if (
        CLEAR_X1 <= panel_x <= CLEAR_X2
        and
        CLEAR_Y1 <= panel_y <= CLEAR_Y2
    ):
        return "CLEAR"


    return None


# ============================================================
# SELECT BUTTON
# ============================================================

def select_button(
    button
):

    global current_color
    global current_color_name


    if button == "RED":

        current_color = RED
        current_color_name = "RED"

        print("Selected RED")


    elif button == "GREEN":

        current_color = GREEN
        current_color_name = "GREEN"

        print("Selected GREEN")


    elif button == "YELLOW":

        current_color = YELLOW
        current_color_name = "YELLOW"

        print("Selected YELLOW")


    elif button == "NAVY":

        current_color = NAVY_BLUE
        current_color_name = "NAVY BLUE"

        print("Selected NAVY BLUE")


    elif button == "CLEAR":

        clear_notepad()


# ============================================================
# CREATE DIGITAL NOTE
# ============================================================

def create_panel():

    panel = np.full(

        (
            HEIGHT,
            PANEL_WIDTH,
            3
        ),

        NOTE_BACKGROUND,

        dtype=np.uint8
    )


    # ========================================================
    # BORDER
    # ========================================================

    cv2.rectangle(

        panel,

        (5, 5),

        (
            PANEL_WIDTH - 6,
            HEIGHT - 6
        ),

        NOTE_BORDER,

        3
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(

        panel,

        "AIRWRITE AI",

        (25, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.95,

        NOTE_TEXT,

        2
    )


    cv2.putText(

        panel,

        "DIGITAL NOTE",

        (25, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        NOTE_GRAY,

        2
    )


    # ========================================================
    # LOCATION
    # ========================================================

    cv2.putText(

        panel,

        "LOCATION",

        (25, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        NOTE_GRAY,

        1
    )


    cv2.putText(

        panel,

        LOCATION_LINE_1,

        (25, 145),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        NOTE_TEXT,

        1
    )


    cv2.putText(

        panel,

        LOCATION_LINE_2,

        (25, 168),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        NOTE_TEXT,

        1
    )


    # ========================================================
    # RECOGNIZED TEXT
    # ========================================================

    display_text = (
        recognized_text
        if recognized_text
        else "_"
    )


    lines = [

        display_text[i:i + 8]

        for i in range(

            0,

            len(display_text),

            8
        )
    ]


    text_y = 220


    for line in lines:

        cv2.putText(

            panel,

            line,

            (25, text_y),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.8,

            NOTE_TEXT,

            3,

            cv2.LINE_AA
        )


        text_y += 70


    # ========================================================
    # CURRENT COLOR
    # ========================================================

    cv2.putText(

        panel,

        "COLOR",

        (25, 350),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        NOTE_GRAY,

        2
    )


    cv2.putText(

        panel,

        current_color_name,

        (25, 390),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        current_color,

        2
    )


    # ========================================================
    # RED BUTTON
    # ========================================================

    cv2.rectangle(

        panel,

        (RED_X1, BUTTON_Y1),

        (RED_X2, BUTTON_Y2),

        RED,

        -1
    )


    cv2.putText(

        panel,

        "RED",

        (
            RED_X1 + 20,
            BUTTON_Y1 + 45
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        (255, 255, 255),

        2
    )


    # ========================================================
    # GREEN BUTTON
    # ========================================================

    cv2.rectangle(

        panel,

        (GREEN_X1, BUTTON_Y1),

        (GREEN_X2, BUTTON_Y2),

        GREEN,

        -1
    )


    cv2.putText(

        panel,

        "GREEN",

        (
            GREEN_X1 + 10,
            BUTTON_Y1 + 45
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        2
    )


    # ========================================================
    # YELLOW BUTTON
    # ========================================================

    cv2.rectangle(

        panel,

        (YELLOW_X1, BUTTON_Y1),

        (YELLOW_X2, BUTTON_Y2),

        YELLOW,

        -1
    )


    cv2.putText(

        panel,

        "YELLOW",

        (
            YELLOW_X1 + 5,
            BUTTON_Y1 + 45
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        (0, 0, 0),

        2
    )


    # ========================================================
    # NAVY BLUE BUTTON
    # ========================================================

    cv2.rectangle(

        panel,

        (NAVY_X1, BUTTON_Y1),

        (NAVY_X2, BUTTON_Y2),

        NAVY_BLUE,

        -1
    )


    cv2.putText(

        panel,

        "NAVY",

        (
            NAVY_X1 + 15,
            BUTTON_Y1 + 45
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (255, 255, 255),

        2
    )


    # ========================================================
    # CLEAR BUTTON
    # ========================================================

    cv2.rectangle(

        panel,

        (CLEAR_X1, CLEAR_Y1),

        (CLEAR_X2, CLEAR_Y2),

        (80, 80, 80),

        -1
    )


    cv2.rectangle(

        panel,

        (CLEAR_X1, CLEAR_Y1),

        (CLEAR_X2, CLEAR_Y2),

        NOTE_BORDER,

        2
    )


    cv2.putText(

        panel,

        "CLEAR",

        (150, CLEAR_Y1 + 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(

        panel,

        "1 = WRITE",

        (25, 635),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        NOTE_TEXT,

        1
    )


    cv2.putText(

        panel,

        "2 = COLOR",

        (25, 665),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        NOTE_TEXT,

        1
    )


    cv2.putText(

        panel,

        "3 = PAUSE | 4 = CLEAR",

        (25, 695),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.43,

        NOTE_GRAY,

        1
    )


    return panel


# ============================================================
# MAIN LOOP
# ============================================================

WINDOW_NAME = "AIRWRITE AI"

cv2.namedWindow(
    WINDOW_NAME
)


try:

    while True:

        # ====================================================
        # CAMERA
        # ====================================================

        success, frame = camera.read()


        if not success:

            print(
                "Camera frame failed."
            )

            break


        frame = cv2.flip(
            frame,
            1
        )


        frame = cv2.resize(

            frame,

            (
                CAMERA_WIDTH,
                HEIGHT
            )
        )


        # ====================================================
        # RGB
        # ====================================================

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb
        )


        timestamp_ms += 33


        # ====================================================
        # FACE LANDMARKER
        # ====================================================

        face_result = (

            face_landmarker.detect_for_video(

                mp_image,

                timestamp_ms
            )
        )


        draw_faces(

            frame,

            face_result
        )


        # ====================================================
        # HAND
        # ====================================================

        hand_result = (

            hand_landmarker.detect_for_video(

                mp_image,

                timestamp_ms
            )
        )


        finger_count = 0

        virtual_cursor = None

        selection_mode = False


        # ====================================================
        # HAND FOUND
        # ====================================================

        if hand_result.hand_landmarks:

            landmarks = (

                hand_result.hand_landmarks[0]
            )


            finger_count = count_fingers(

                landmarks
            )


            # =================================================
            # HAND SKELETON
            # =================================================

            connections = [

                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (5, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (9, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (13, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (0, 17)
            ]


            for a, b in connections:

                x1 = int(
                    landmarks[a].x
                    *
                    CAMERA_WIDTH
                )


                y1 = int(
                    landmarks[a].y
                    *
                    HEIGHT
                )


                x2 = int(
                    landmarks[b].x
                    *
                    CAMERA_WIDTH
                )


                y2 = int(
                    landmarks[b].y
                    *
                    HEIGHT
                )


                cv2.line(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 255),

                    2
                )


            # =================================================
            # LANDMARKS
            # =================================================

            for landmark in landmarks:

                x = int(
                    landmark.x
                    *
                    CAMERA_WIDTH
                )


                y = int(
                    landmark.y
                    *
                    HEIGHT
                )


                cv2.circle(

                    frame,

                    (x, y),

                    3,

                    GREEN,

                    -1
                )


            # =================================================
            # INDEX FINGER
            # =================================================

            index = landmarks[8]


            finger_x = int(
                index.x
                *
                CAMERA_WIDTH
            )


            finger_y = int(
                index.y
                *
                HEIGHT
            )


            # =================================================
            # 2 FINGERS = SELECT COLOR
            # =================================================

            if finger_count == 2:

                selection_mode = True

                writing = False

                previous_point = None

                smooth_x = None
                smooth_y = None


                panel_x = int(

                    (

                        finger_x

                        /

                        CAMERA_WIDTH

                    )

                    *

                    PANEL_WIDTH
                )


                panel_y = finger_y


                virtual_cursor = (

                    panel_x,

                    panel_y
                )


                button = get_button(

                    panel_x,

                    panel_y
                )


                cv2.circle(

                    frame,

                    (
                        finger_x,
                        finger_y
                    ),

                    15,

                    (255, 255, 255),

                    3
                )


                if button is not None:

                    cv2.putText(

                        frame,

                        f"SELECT: {button}",

                        (20, 50),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.8,

                        (255, 255, 255),

                        3
                    )


                    if selection_target != button:

                        selection_target = button

                        selection_start_time = (

                            time.time()
                        )


                    else:

                        elapsed = (

                            time.time()

                            -

                            selection_start_time
                        )


                        progress = min(

                            elapsed

                            /

                            SELECT_HOLD_TIME,

                            1.0
                        )


                        cv2.circle(

                            frame,

                            (
                                finger_x,
                                finger_y
                            ),

                            25,

                            (255, 255, 255),

                            3
                        )


                        if progress >= 1.0:

                            select_button(

                                button
                            )


                            selection_target = None

                            selection_start_time = None


                else:

                    selection_target = None

                    selection_start_time = None


            # =================================================
            # 1 FINGER = WRITE
            # =================================================

            elif finger_count == 1:

                four_finger_clear_done = False

                selection_target = None

                selection_start_time = None

                writing = True


                cv2.putText(

                    frame,

                    "WRITING",

                    (20, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    current_color,

                    3
                )


                if smooth_x is None:

                    smooth_x = finger_x
                    smooth_y = finger_y


                else:

                    smooth_x = int(

                        SMOOTHING
                        *
                        smooth_x

                        +

                        (1 - SMOOTHING)
                        *
                        finger_x
                    )


                    smooth_y = int(

                        SMOOTHING
                        *
                        smooth_y

                        +

                        (1 - SMOOTHING)
                        *
                        finger_y
                    )


                current_point = (

                    smooth_x,

                    smooth_y
                )


                cv2.circle(

                    frame,

                    current_point,

                    10,

                    current_color,

                    -1
                )


                if previous_point is not None:

                    distance = np.sqrt(

                        (

                            current_point[0]

                            -

                            previous_point[0]

                        ) ** 2

                        +

                        (

                            current_point[1]

                            -

                            previous_point[1]

                        ) ** 2
                    )


                    if distance > 1:

                        cv2.line(

                            color_canvas,

                            previous_point,

                            current_point,

                            current_color,

                            BRUSH_SIZE,

                            cv2.LINE_AA
                        )


                        cv2.line(

                            gray_canvas,

                            previous_point,

                            current_point,

                            255,

                            BRUSH_SIZE,

                            cv2.LINE_AA
                        )


                        last_draw_time = (

                            time.time()
                        )


                        letter_ready = True


                previous_point = current_point


            # =================================================
            # 3 FINGERS = PAUSE
            # =================================================

            elif finger_count == 3:

                writing = False

                previous_point = None

                smooth_x = None
                smooth_y = None

                selection_target = None
                selection_start_time = None

                four_finger_clear_done = False


                cv2.putText(

                    frame,

                    "PAUSED",

                    (20, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (255, 255, 255),

                    3
                )


            # =================================================
            # 4 FINGERS = CLEAR NOTEPAD
            # =================================================

            elif finger_count == 4:

                writing = False

                previous_point = None

                smooth_x = None
                smooth_y = None

                selection_target = None
                selection_start_time = None


                if not four_finger_clear_done:

                    clear_notepad()

                    four_finger_clear_done = True


                cv2.putText(

                    frame,

                    "NOTEPAD CLEARED",

                    (20, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (255, 255, 255),

                    3
                )


            # =================================================
            # OTHER
            # =================================================

            else:

                writing = False

                previous_point = None

                smooth_x = None
                smooth_y = None

                selection_target = None
                selection_start_time = None

                four_finger_clear_done = False


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            writing = False

            previous_point = None

            smooth_x = None
            smooth_y = None

            selection_target = None
            selection_start_time = None

            four_finger_clear_done = False


        # ====================================================
        # RECOGNIZE LETTER
        # ====================================================

        if (

            letter_ready

            and

            not writing

            and

            not selection_mode

            and

            (

                time.time()

                -

                last_draw_time

            ) >= LETTER_PAUSE

        ):

            letter, confidence = (

                recognize_letter()
            )


            if letter is not None:

                print(
                    "Detected letter:",
                    letter
                )


                print(
                    "Confidence:",
                    round(
                        confidence * 100,
                        2
                    ),
                    "%"
                )


                if confidence >= 0.20:

                    recognized_text += letter


            clear_drawing()


        # ====================================================
        # CAMERA DRAWING
        # ====================================================

        frame = cv2.addWeighted(

            frame,

            1.0,

            color_canvas,

            0.8,

            0
        )


        # ====================================================
        # DIGITAL NOTE
        # ====================================================

        panel = create_panel()


        # ====================================================
        # VIRTUAL CURSOR
        # ====================================================

        if virtual_cursor is not None:

            px, py = virtual_cursor


            cv2.circle(

                panel,

                (px, py),

                14,

                (0, 0, 0),

                3
            )


        # ====================================================
        # COMBINE
        # ====================================================

        combined = np.hstack(

            (

                frame,

                panel

            )
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(

            WINDOW_NAME,

            combined
        )


        # ====================================================
        # EXIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if (

            key == ord("q")

            or

            key == 27

        ):

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    camera.release()

    hand_landmarker.close()

    face_landmarker.close()

    cv2.destroyAllWindows()


print()

print("==============================================")

print("           AIRWRITE AI STOPPED")

print("==============================================")

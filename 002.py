from pathlib import Path

import cv2
import mediapipe as mp

vision = mp.tasks.vision
model_path = Path(__file__).resolve().parent / "model" / "hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
)

connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
landmark_style = vision.drawing_styles.get_default_hand_landmarks_style()
connection_style = vision.drawing_styles.get_default_hand_connections_style()


def draw_landmarks_on_image(bgr_image, result):
    annotated_image = bgr_image.copy()
    height, width = annotated_image.shape[:2]

    for hand_landmarks, handedness in zip(
        result.hand_landmarks, result.handedness
    ):
        vision.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            connections,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style,
        )

        if handedness:
            label = handedness[0].category_name
            text_x = max(0, int(min(point.x for point in hand_landmarks) * width))
            text_y = max(
                25, int(min(point.y for point in hand_landmarks) * height) - 10
            )

            cv2.putText(
                annotated_image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (88, 205, 54),
                1,
                cv2.LINE_AA,
            )

    return annotated_image


cap = cv2.VideoCapture(0)

try:
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            annotated_image = draw_landmarks_on_image(frame, result)
            cv2.imshow("oxxostudio", annotated_image)

            if (cv2.waitKey(5) & 0xFF) == ord("q"):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
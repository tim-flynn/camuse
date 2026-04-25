from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np


WINDOW_NAME = "Camuse tracker"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class HandState:
    label: str
    score: float
    wrist: Point
    index_tip: Point
    palm_center: Point


@dataclass(frozen=True)
class MouthState:
    openness: float
    upper_lip: Point
    lower_lip: Point
    center: Point


@dataclass(frozen=True)
class FrameState:
    timestamp: float
    hands: list[HandState]
    mouth: MouthState | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track hand position and mouth openness from a webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to open.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum hands to track.")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirror view. By default the image behaves like a mirror.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print one JSON state line per processed frame.",
    )
    return parser.parse_args()


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def landmark_to_point(landmark: object) -> Point:
    return Point(float(landmark.x), float(landmark.y))


def average_points(points: Iterable[Point]) -> Point:
    values = list(points)
    return Point(
        sum(point.x for point in values) / len(values),
        sum(point.y for point in values) / len(values),
    )


def pixel(point: Point, width: int, height: int) -> tuple[int, int]:
    return (
        int(np.clip(point.x, 0.0, 1.0) * width),
        int(np.clip(point.y, 0.0, 1.0) * height),
    )


def extract_hands(hand_results: object) -> list[HandState]:
    landmarks = hand_results.multi_hand_landmarks or []
    handedness = hand_results.multi_handedness or []
    hands: list[HandState] = []

    for hand_landmarks, hand_info in zip(landmarks, handedness):
        points = [landmark_to_point(landmark) for landmark in hand_landmarks.landmark]
        palm_center = average_points(
            [
                points[0],
                points[5],
                points[9],
                points[13],
                points[17],
            ]
        )
        classification = hand_info.classification[0]
        hands.append(
            HandState(
                label=classification.label,
                score=float(classification.score),
                wrist=points[0],
                index_tip=points[8],
                palm_center=palm_center,
            )
        )

    return hands


def extract_mouth(face_results: object) -> MouthState | None:
    faces = face_results.multi_face_landmarks or []
    if not faces:
        return None

    points = [landmark_to_point(landmark) for landmark in faces[0].landmark]
    upper_lip = points[13]
    lower_lip = points[14]
    left_corner = points[61]
    right_corner = points[291]
    mouth_width = max(distance(left_corner, right_corner), 1e-6)
    openness = distance(upper_lip, lower_lip) / mouth_width

    return MouthState(
        openness=float(openness),
        upper_lip=upper_lip,
        lower_lip=lower_lip,
        center=average_points([upper_lip, lower_lip, left_corner, right_corner]),
    )


def draw_hands(
    frame: np.ndarray,
    hand_results: object,
    hands: list[HandState],
    drawing_utils: object,
    hand_connections: object,
) -> None:
    height, width = frame.shape[:2]
    landmarks = hand_results.multi_hand_landmarks or []

    for hand_landmarks in landmarks:
        drawing_utils.draw_landmarks(frame, hand_landmarks, hand_connections)

    for hand in hands:
        palm = pixel(hand.palm_center, width, height)
        index_tip = pixel(hand.index_tip, width, height)
        cv2.circle(frame, palm, 12, (0, 180, 255), -1)
        cv2.circle(frame, index_tip, 8, (80, 220, 80), -1)
        cv2.line(frame, palm, index_tip, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"{hand.label} palm {hand.palm_center.x:.2f}, {hand.palm_center.y:.2f}",
            (palm[0] + 12, palm[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_mouth(frame: np.ndarray, mouth: MouthState | None) -> None:
    if mouth is None:
        cv2.putText(
            frame,
            "Mouth: not detected",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (60, 60, 255),
            2,
            cv2.LINE_AA,
        )
        return

    height, width = frame.shape[:2]
    upper = pixel(mouth.upper_lip, width, height)
    lower = pixel(mouth.lower_lip, width, height)
    center = pixel(mouth.center, width, height)

    cv2.line(frame, upper, lower, (255, 180, 40), 3)
    cv2.circle(frame, center, 7, (255, 180, 40), -1)
    cv2.putText(
        frame,
        f"Mouth open: {mouth.openness:.3f}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status(frame: np.ndarray, fps: float, hand_count: int) -> None:
    height = frame.shape[0]
    cv2.putText(
        frame,
        f"Hands: {hand_count}   FPS: {fps:.1f}   Press q or Esc to quit",
        (20, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def print_state(state: FrameState) -> None:
    print(json.dumps(asdict(state), separators=(",", ":")), flush=True)


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Check Windows camera permissions or try --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def run_tracker(args: argparse.Namespace) -> None:
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    drawing_utils = mp.solutions.drawing_utils

    capture = open_camera(args.camera, args.width, args.height)
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        with (
            mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=args.max_hands,
                model_complexity=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ) as hands_model,
            mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            ) as face_model,
        ):
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames.")

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                hand_results = hands_model.process(rgb_frame)
                face_results = face_model.process(rgb_frame)
                rgb_frame.flags.writeable = True

                hands = extract_hands(hand_results)
                mouth = extract_mouth(face_results)
                state = FrameState(time.time(), hands, mouth)

                now = time.perf_counter()
                elapsed = max(now - previous_time, 1e-6)
                previous_time = now
                fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)

                draw_hands(
                    frame,
                    hand_results,
                    hands,
                    drawing_utils,
                    mp_hands.HAND_CONNECTIONS,
                )
                draw_mouth(frame, mouth)
                draw_status(frame, fps, len(hands))

                if args.print_json:
                    print_state(state)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    try:
        run_tracker(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

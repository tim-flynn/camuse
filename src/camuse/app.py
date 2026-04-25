from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import cv2
import mediapipe as mp
import mido
import numpy as np


WINDOW_NAME = "Camuse tracker"
MIDI_CC_MIN = 0
MIDI_CC_MAX = 127
MIDI_INPUT_DEFINITIONS = (
    ("palm_x", "Palm X", "cc_palm_x"),
    ("palm_y", "Palm Y", "cc_palm_y"),
    ("index_x", "Index X", "cc_index_x"),
    ("index_y", "Index Y", "cc_index_y"),
    ("mouth", "Mouth", "cc_mouth"),
)


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


@dataclass(frozen=True)
class MidiConfig:
    port: str | None
    channel: int
    control_hand: str
    smoothing: float
    deadband: int
    mouth_open_max: float
    cc_palm_x: int
    cc_palm_y: int
    cc_index_x: int
    cc_index_y: int
    cc_mouth: int


def cc_number(value: str) -> int:
    number = int(value)
    if number != -1 and not MIDI_CC_MIN <= number <= MIDI_CC_MAX:
        raise argparse.ArgumentTypeError("MIDI CC must be -1 or between 0 and 127.")
    return number


def midi_channel(value: str) -> int:
    channel = int(value)
    if not 1 <= channel <= 16:
        raise argparse.ArgumentTypeError("MIDI channel must be between 1 and 16.")
    return channel


def normalized_float(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0.0 and 1.0.")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be 0 or greater.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("Value must be greater than 0.")
    return number


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
    parser.add_argument(
        "--list-midi-ports",
        action="store_true",
        help="List available MIDI output ports and exit.",
    )
    parser.add_argument(
        "--midi-port",
        help=(
            "MIDI output port to send CC values to. On Windows this is usually "
            "a virtual MIDI port from a tool such as loopMIDI."
        ),
    )
    parser.add_argument(
        "--midi-channel",
        type=midi_channel,
        default=1,
        help="MIDI channel to send CC values on, from 1 to 16.",
    )
    parser.add_argument(
        "--control-hand",
        choices=("first", "Left", "Right"),
        default="first",
        help="Which tracked hand should drive hand MIDI CC values.",
    )
    parser.add_argument(
        "--midi-smoothing",
        type=normalized_float,
        default=0.35,
        help="MIDI smoothing amount. 1.0 follows tracking immediately; lower is smoother.",
    )
    parser.add_argument(
        "--midi-deadband",
        type=nonnegative_int,
        default=1,
        help="Minimum CC value change before sending another message.",
    )
    parser.add_argument(
        "--mouth-open-max",
        type=positive_float,
        default=0.45,
        help="Mouth openness value that maps to MIDI 127.",
    )
    parser.add_argument(
        "--cc-palm-x",
        type=cc_number,
        default=20,
        help="CC for palm horizontal position. Use -1 to disable.",
    )
    parser.add_argument(
        "--cc-palm-y",
        type=cc_number,
        default=21,
        help="CC for palm vertical position. Use -1 to disable.",
    )
    parser.add_argument(
        "--cc-index-x",
        type=cc_number,
        default=22,
        help="CC for index fingertip horizontal position. Use -1 to disable.",
    )
    parser.add_argument(
        "--cc-index-y",
        type=cc_number,
        default=23,
        help="CC for index fingertip vertical position. Use -1 to disable.",
    )
    parser.add_argument(
        "--cc-mouth",
        type=cc_number,
        default=24,
        help="CC for mouth openness. Use -1 to disable.",
    )
    return parser.parse_args()


def midi_config_from_args(args: argparse.Namespace) -> MidiConfig:
    return MidiConfig(
        port=args.midi_port,
        channel=args.midi_channel,
        control_hand=args.control_hand,
        smoothing=args.midi_smoothing,
        deadband=args.midi_deadband,
        mouth_open_max=args.mouth_open_max,
        cc_palm_x=args.cc_palm_x,
        cc_palm_y=args.cc_palm_y,
        cc_index_x=args.cc_index_x,
        cc_index_y=args.cc_index_y,
        cc_mouth=args.cc_mouth,
    )


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


def midi_value(value: float) -> int:
    return int(round(float(np.clip(value, 0.0, 1.0)) * MIDI_CC_MAX))


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
        f"Hands: {hand_count}   FPS: {fps:.1f}   1-5 toggle MIDI inputs   q/Esc quits",
        (20, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_midi_status(
    frame: np.ndarray,
    midi_controller: MidiController | None,
    midi_controls: MidiInputControls,
) -> None:
    height, width = frame.shape[:2]
    input_states = "   ".join(
        f"{index}: {label} {midi_controls.status_label(name)}"
        for index, (name, label, _) in enumerate(MIDI_INPUT_DEFINITIONS, start=1)
    )
    output = midi_controller.port_name if midi_controller is not None else "no output"
    text = f"MIDI -> {output}   {input_states}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    text_width = cv2.getTextSize(text, font, scale, thickness)[0][0]
    max_width = max(width - 40, 1)
    if text_width > max_width:
        scale = max(0.35, scale * (max_width / text_width))

    cv2.putText(
        frame,
        text,
        (20, height - 56),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def print_state(state: FrameState) -> None:
    print(json.dumps(asdict(state), separators=(",", ":")), flush=True)


def list_midi_ports() -> None:
    names = mido.get_output_names()
    if not names:
        print("No MIDI output ports found.")
        return

    print("MIDI output ports:")
    for name in names:
        print(f"- {name}")


def resolve_midi_port(port: str) -> str:
    names = mido.get_output_names()
    if port in names:
        return port

    matches = [name for name in names if port.lower() in name.lower()]
    if len(matches) == 1:
        return matches[0]

    if matches:
        choices = "\n".join(f"- {name}" for name in matches)
        raise RuntimeError(
            f'MIDI port "{port}" matched multiple output ports:\n{choices}\n'
            "Use the full port name."
        )

    available = "\n".join(f"- {name}" for name in names) or "- none"
    raise RuntimeError(
        f'MIDI output port "{port}" was not found.\n'
        f"Available output ports:\n{available}"
    )


def select_control_hand(hands: list[HandState], preference: str) -> HandState | None:
    if not hands:
        return None
    if preference == "first":
        return hands[0]

    return next((hand for hand in hands if hand.label == preference), None)


class MidiInputControls:
    def __init__(self, config: MidiConfig) -> None:
        self.available = {
            name: getattr(config, cc_attribute) != -1
            for name, _, cc_attribute in MIDI_INPUT_DEFINITIONS
        }
        self.enabled = {
            name: self.available[name] for name, _, _ in MIDI_INPUT_DEFINITIONS
        }
        self.trackbars: dict[str, str] = {}

    def create(self) -> None:
        cv2.namedWindow(WINDOW_NAME)
        for name, label, _ in MIDI_INPUT_DEFINITIONS:
            value = int(self.enabled[name])
            trackbar = f"Send {label}"
            self.trackbars[name] = trackbar
            cv2.createTrackbar(
                trackbar,
                WINDOW_NAME,
                value,
                1,
                self._set_enabled_callback(name),
            )

    def is_enabled(self, name: str) -> bool:
        return self.available.get(name, False) and self.enabled.get(name, False)

    def toggle_by_index(self, index: int) -> None:
        if not 0 <= index < len(MIDI_INPUT_DEFINITIONS):
            return

        name = MIDI_INPUT_DEFINITIONS[index][0]
        if not self.available[name]:
            return

        self.enabled[name] = not self.enabled[name]
        cv2.setTrackbarPos(
            self.trackbars[name],
            WINDOW_NAME,
            int(self.enabled[name]),
        )

    def status_label(self, name: str) -> str:
        if not self.available.get(name, False):
            return "unmapped"
        return "on" if self.enabled.get(name, False) else "off"

    def _set_enabled_callback(self, name: str) -> Callable[[int], None]:
        def set_enabled(value: int) -> None:
            if not self.available[name]:
                self.enabled[name] = False
                if value != 0:
                    cv2.setTrackbarPos(self.trackbars[name], WINDOW_NAME, 0)
                return

            self.enabled[name] = value == 1

        return set_enabled


class MidiController:
    def __init__(self, config: MidiConfig) -> None:
        if config.port is None:
            raise ValueError("A MIDI port is required.")

        self.config = config
        self.port_name = resolve_midi_port(config.port)
        self.output = mido.open_output(self.port_name)
        self.channel = config.channel - 1
        self.smoothed_values: dict[int, float] = {}
        self.sent_values: dict[int, int] = {}

    def close(self) -> None:
        self.output.close()

    def send(self, cc: int, value: float) -> None:
        if cc == -1:
            return

        target = midi_value(value)
        previous = self.smoothed_values.get(cc, float(target))
        smoothed = previous + ((target - previous) * self.config.smoothing)
        quantized = int(round(smoothed))
        last_sent = self.sent_values.get(cc)

        self.smoothed_values[cc] = smoothed
        if last_sent is not None and abs(quantized - last_sent) < self.config.deadband:
            return

        message = mido.Message(
            "control_change",
            channel=self.channel,
            control=cc,
            value=quantized,
        )
        self.output.send(message)
        self.sent_values[cc] = quantized

    def update(
        self,
        state: FrameState,
        input_controls: MidiInputControls | None = None,
    ) -> None:
        def enabled(name: str) -> bool:
            return input_controls is None or input_controls.is_enabled(name)

        hand = select_control_hand(state.hands, self.config.control_hand)
        if hand is not None:
            if enabled("palm_x"):
                self.send(self.config.cc_palm_x, hand.palm_center.x)
            if enabled("palm_y"):
                self.send(self.config.cc_palm_y, 1.0 - hand.palm_center.y)
            if enabled("index_x"):
                self.send(self.config.cc_index_x, hand.index_tip.x)
            if enabled("index_y"):
                self.send(self.config.cc_index_y, 1.0 - hand.index_tip.y)

        if state.mouth is not None and enabled("mouth"):
            self.send(
                self.config.cc_mouth,
                state.mouth.openness / self.config.mouth_open_max,
            )


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

    midi_config = midi_config_from_args(args)
    midi_controller: MidiController | None = None
    midi_controls = MidiInputControls(midi_config)
    capture: cv2.VideoCapture | None = None
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        midi_controls.create()
        midi_controller = MidiController(midi_config) if midi_config.port else None
        capture = open_camera(args.camera, args.width, args.height)

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
                draw_midi_status(frame, midi_controller, midi_controls)

                if args.print_json:
                    print_state(state)
                if midi_controller is not None:
                    midi_controller.update(state, midi_controls)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if ord("1") <= key <= ord("5"):
                    midi_controls.toggle_by_index(key - ord("1"))
    finally:
        if midi_controller is not None:
            midi_controller.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    try:
        if args.list_midi_ports:
            list_midi_ports()
            return
        run_tracker(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

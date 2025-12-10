#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose and facial expression detection.

This module provides a config-driven emoji reaction system that detects user poses
and facial expressions using MediaPipe, then displays corresponding emoji images.

Classes:
    EmojiConfig: Configuration holder for application settings and state definitions
    ImageManager: Handles loading and management of emoji images
    StateDetector: Detects user state based on pose and facial landmarks
    EmojiReactor: Main application orchestrator

Adding new states:
    1. Add state definition to EmojiConfig.states list with priority
    2. Add detection logic to StateDetector (if needed)
    3. Add corresponding emoji image file to project directory
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe solutions (used by StateDetector)
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class EmojiConfig:
    """Configuration holder for the Emoji Reactor application."""

    def __init__(self):
        # Window settings
        self.window_width = 720
        self.window_height = 450

        # Detection thresholds
        self.smile_threshold = 0.25
        self.min_detection_confidence = 0.5
        self.min_tracking_confidence = 0.5

        # State definitions with priority (higher = higher priority)
        # Each state: name, image_file, emoji_unicode, priority, detection_type
        self.states = [
            {
                'name': 'HANDS_UP',
                'image_file': 'aircat.png',
                'emoji': '🙌',
                'priority': 40,
                'detector': 'pose'
            },
            {
                'name': 'ONE_HAND_UP',
                'image_file': 'hmm.png',
                'emoji': '✋',
                'priority': 30,
                'detector': 'pose'
            },
            {
                'name': 'SMILING',
                'image_file': 'catxd.png',
                'emoji': '😊',
                'priority': 20,
                'detector': 'face'
            },
            {
                'name': 'STRAIGHT_FACE',
                'image_file': 'toletole.png',
                'emoji': '😐',
                'priority': 10,
                'detector': 'face'
            }
        ]

    def get_state_by_name(self, name):
        """Get state config by name."""
        return next((s for s in self.states if s['name'] == name), None)

    def get_sorted_states(self):
        """Get states sorted by priority (highest first)."""
        return sorted(self.states, key=lambda x: x['priority'], reverse=True)


class ImageManager:
    """Manages loading and resizing of emoji images."""

    def __init__(self, config):
        self.config = config
        self.images = {}  # name -> image mapping
        self.blank_image = None

    def load_images(self):
        """Load and resize all emoji images based on config."""
        for state in self.config.states:
            name = state['name']
            file_path = state['image_file']

            image = cv2.imread(file_path)
            if image is None:
                raise FileNotFoundError(f"Could not load {file_path} for state {name}")

            # Resize to window dimensions
            resized = cv2.resize(
                image,
                (self.config.window_width, self.config.window_height)
            )
            self.images[name] = resized

        # Create blank image for unknown states
        self.blank_image = np.zeros(
            (self.config.window_height, self.config.window_width, 3),
            dtype=np.uint8
        )

    def get_image(self, state_name):
        """Get image for a given state name."""
        return self.images.get(state_name, self.blank_image)


class StateDetector:
    """Detects current state based on pose and face landmarks."""

    def __init__(self, config):
        self.config = config
        self.mp_pose = mp_pose
        self.mp_face_mesh = mp_face_mesh

        # Initialize MediaPipe models
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=config.min_detection_confidence
        )

    def detect_state(self, frame_rgb):
        """
        Detect current state from RGB frame.
        Returns: state_name (str)
        """
        detected_states = []

        # Check pose-based states
        pose_state = self._check_pose_state(frame_rgb)
        if pose_state:
            detected_states.append(pose_state)

        # Check face-based states (only if no high-priority pose detected)
        if not self._has_high_priority_pose(pose_state):
            face_state = self._check_face_state(frame_rgb)
            if face_state:
                detected_states.append(face_state)

        # Return highest priority detected state, or default
        return self._get_highest_priority_state(detected_states)

    def _check_pose_state(self, frame_rgb):
        """Check for hands-up states."""
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_index = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX]
        right_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX]

        # Check both hands up
        if (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y):
            return 'HANDS_UP'

        # Check one hand up
        if (left_index.y < left_shoulder.y) or (right_index.y < right_shoulder.y):
            return 'ONE_HAND_UP'

        return None

    def _check_face_state(self, frame_rgb):
        """Check for smile vs straight face."""
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return 'STRAIGHT_FACE'  # Default

        for face_landmarks in results.multi_face_landmarks:
            # Landmark indices for mouth corners and lips
            left_corner = face_landmarks.landmark[291]
            right_corner = face_landmarks.landmark[61]
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            # Calculate mouth aspect ratio
            mouth_width = ((right_corner.x - left_corner.x)**2 +
                          (right_corner.y - left_corner.y)**2)**0.5
            mouth_height = ((lower_lip.x - upper_lip.x)**2 +
                           (lower_lip.y - upper_lip.y)**2)**0.5

            if mouth_width > 0:
                mouth_aspect_ratio = mouth_height / mouth_width
                if mouth_aspect_ratio > self.config.smile_threshold:
                    return 'SMILING'

        return 'STRAIGHT_FACE'

    def _has_high_priority_pose(self, pose_state):
        """Check if pose state has higher priority than face states."""
        if not pose_state:
            return False
        pose_config = self.config.get_state_by_name(pose_state)
        return pose_config and pose_config['priority'] >= 30  # Above face states

    def _get_highest_priority_state(self, detected_states):
        """Return highest priority state from detected states."""
        if not detected_states:
            return 'STRAIGHT_FACE'  # Default fallback

        state_priorities = []
        for state_name in detected_states:
            state_config = self.config.get_state_by_name(state_name)
            if state_config:
                state_priorities.append((state_name, state_config['priority']))

        if not state_priorities:
            return 'STRAIGHT_FACE'

        # Return state with highest priority
        return max(state_priorities, key=lambda x: x[1])[0]

    def cleanup(self):
        """Release MediaPipe resources."""
        self.pose.close()
        self.face_mesh.close()


class EmojiReactor:
    """Main application class that orchestrates the emoji reactor."""

    def __init__(self, config):
        self.config = config
        self.image_manager = ImageManager(config)
        self.state_detector = StateDetector(config)
        self.cap = None

    def initialize(self):
        """Initialize all components."""
        # Load images
        try:
            self.image_manager.load_images()
        except FileNotFoundError as e:
            print(f"Error loading emoji images: {e}")
            self._print_expected_files()
            return False

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return False

        # Setup windows
        self._setup_windows()

        return True

    def _setup_windows(self):
        """Create and position display windows."""
        cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Feed', self.config.window_width, self.config.window_height)
        cv2.resizeWindow('Emoji Output', self.config.window_width, self.config.window_height)
        cv2.moveWindow('Camera Feed', 100, 100)
        cv2.moveWindow('Emoji Output', self.config.window_width + 150, 100)


    def _print_expected_files(self):
        """Print expected emoji files."""
        print("\nExpected files:")
        for state in self.config.states:
            print(f"- {state['image_file']} ({state['emoji']} {state['name']})")

    def run(self):
        """Main application loop."""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            # Flip and convert frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # Detect current state
            current_state = self.state_detector.detect_state(frame_rgb)

            # Get state config and image
            state_config = self.config.get_state_by_name(current_state)
            emoji_image = self.image_manager.get_image(current_state)
            emoji_name = state_config['emoji'] if state_config else '❓'

            # Prepare camera display
            camera_frame = self._prepare_camera_display(frame, current_state, emoji_name)

            # Show windows
            cv2.imshow('Camera Feed', camera_frame)
            cv2.imshow('Emoji Output', emoji_image)

            # Check for quit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    def _prepare_camera_display(self, frame, state_name, emoji_name):
        """Add overlay text to camera frame."""
        frame_resized = cv2.resize(frame, (self.config.window_width, self.config.window_height))

        cv2.putText(
            frame_resized,
            f'STATE: {state_name} {emoji_name}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame_resized,
            'Press "q" to quit',
            (10, self.config.window_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame_resized

    def cleanup(self):
        """Release all resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.state_detector.cleanup()


def main():
    """Entry point for the emoji reactor application."""
    config = EmojiConfig()
    reactor = EmojiReactor(config)

    if not reactor.initialize():
        return 1

    try:
        reactor.run()
    finally:
        reactor.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())

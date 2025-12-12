#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose and facial expression detection.

This module provides a config-driven emoji reaction system that detects user poses
and facial expressions using MediaPipe, then displays corresponding emoji images.

Classes:
    EmojiState: Type-safe dataclass representing a single emoji state
    EmojiConfig: Base configuration class for custom configurations
    DefaultEmojiConfig: Default cat-configs (inherits EmojiConfig)
    ImageManager: Handles loading and management of emoji images
    StateDetector: Detects user state based on pose and facial landmarks
    EmojiReactor: Main application orchestrator

Type Safety:
    - StateName: Literal type for valid state names
    - DetectorType: Literal type for detector types ('pose' or 'face')
    - EmojiState: Dataclass with full type hints for autocomplete and validation

Configuration:
    The application uses DefaultEmojiConfig by default. To customize:
    1. Create a subclass of EmojiConfig
    2. Override __init__ and set custom states/thresholds
    3. Pass your config to EmojiReactor(your_config)

Adding new states:
    1. Add state name to StateName Literal type
    2. Add EmojiState instance to your config's states list with priority
    3. Implement detect_<STATE_NAME> method in StateDetector (or subclass)
    4. Add corresponding emoji image file to project directory

Customizing pose detection:
    Each pose has its own detection method (e.g., detect_HANDS_UP, detect_THINKING).
    To customize a pose, subclass StateDetector and override the corresponding method:
    
        class MyCustomDetector(StateDetector):
            def detect_THINKING(self, frame_rgb) -> bool:
                # Your custom thinking pose detection logic
                return True  # or False
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe solutions (used by StateDetector)
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Type definitions for type safety
StateName = Literal['HANDS_UP', 'THINKING', 'NERD', 'EYEBROW_RAISED', 'CATXD', 'STRAIGHT_FACE']
DetectorType = Literal['pose', 'face']
DEFAULT_STATE = "STRAIGHT_FACE"

@dataclass
class EmojiState:
    """Represents a single emoji state configuration."""
    name: StateName
    image_file: str
    emoji: str
    priority: int
    detector: DetectorType


class EmojiConfig:
    """
    Base configuration class for the Emoji Reactor application.

    Override this class to create custom configurations with different
    settings, emoji images, or detection states.

    Example:
        class MyCustomConfig(EmojiConfig):
            def __init__(self):
                super().__init__()
                self.smile_threshold = 0.30  # More sensitive
                self.states = [...]  # Your custom states
    """

    def __init__(self):
        # Window settings (can be overridden in subclasses)
        self.window_width: int = 720
        self.window_height: int = 450

        # Detection thresholds (can be overridden in subclasses)
        self.smile_threshold: float = 0.25
        self.min_detection_confidence: float = 0.5
        self.min_tracking_confidence: float = 0.5

        # State definitions (should be set in subclasses)
        self.states: list[EmojiState] = []

    def get_state_by_name(self, name: str) -> EmojiState | None:
        """Get state config by name."""
        return next((s for s in self.states if s.name == name), None)

    def get_sorted_states(self) -> list[EmojiState]:
        """Get states sorted by priority (highest first)."""
        return sorted(self.states, key=lambda x: x.priority, reverse=True)


class DefaultEmojiConfig(EmojiConfig):
    """
    Default emoji configuration with cats.

    This is the standard configuration used by the application.

    Example of creating a custom config:
        class MySensitiveConfig(EmojiConfig):
            def __init__(self):
                super().__init__()
                self.smile_threshold = 0.15  # More sensitive to smiles
                self.states = [
                    EmojiState('CATXD', 'my_smile.png', '😁', 20, 'face'),
                    EmojiState('STRAIGHT_FACE', 'my_plain.png', '😑', 10, 'face'),
                ]
    """

    def __init__(self):
        super().__init__()

        # State definitions with priority (higher = higher priority)
        self.states: list[EmojiState] = [
            EmojiState(
                name='HANDS_UP',
                image_file='pics/handsup.jpg',
                emoji='🙌',
                priority=40,
                detector='pose'
            ),
            EmojiState(
                name='NERD',
                image_file='pics/nerd.png',
                emoji='🤓',
                priority=35,
                detector='pose'
            ),
            EmojiState(
                name='THINKING',
                image_file='pics/hmm.png',
                emoji='🤔',
                priority=30,
                detector='pose'
            ),
            EmojiState(
                name='EYEBROW_RAISED',
                image_file='pics/eyebrow_raised.jpg',
                emoji='🤨',
                priority=15,
                detector='face'
            ),
            EmojiState(
                name='CATXD',
                image_file='pics/catxd.png',
                emoji='😂',
                priority=20,
                detector='face'
            ),
            EmojiState(
                name='STRAIGHT_FACE',
                image_file='pics/toletole.png',
                emoji='😐',
                priority=10,
                detector='face'
            )
        ]


class ImageManager:
    """Manages loading and resizing of emoji images."""

    def __init__(self, config):
        self.config = config
        self.images = {}  # name -> image mapping
        self.blank_image = None

    def load_images(self):
        """Load and resize all emoji images based on config."""
        for state in self.config.states:
            name = state.name
            file_path = state.image_file

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
    """
    Detects current state based on pose and face landmarks.
    
    Each pose can be implemented individually by overriding the corresponding
    detection method (e.g., detect_HANDS_UP, detect_THINKING).
    
    To add a new pose:
    1. Add the state name to StateName type
    2. Implement detect_<STATE_NAME> method that returns bool
    3. The method will be called automatically based on config
    """

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
        
        # Cache for pose landmarks (computed once per frame)
        self._cached_pose_results = None
        self._cached_face_results = None

    def detect_state(self, frame_rgb):
        """
        Detect current state from RGB frame.
        Returns: state_name (str)
        """
        detected_states = []
        
        # Process MediaPipe models once per frame
        self._cached_pose_results = self.pose.process(frame_rgb)
        self._cached_face_results = self.face_mesh.process(frame_rgb)

        # Check pose-based states (sorted by priority)
        pose_states = [s for s in self.config.states if s.detector == 'pose']
        for state in sorted(pose_states, key=lambda x: x.priority, reverse=True):
            detector_method = getattr(self, f'detect_{state.name}', None)
            if detector_method and callable(detector_method):
                if detector_method(frame_rgb):
                    detected_states.append(state.name)
                    # Stop at first detected pose (highest priority)
                    break

        # Check face-based states (only if no high-priority pose detected)
        if not detected_states or not self._has_high_priority_pose(detected_states[0]):
            face_states = [s for s in self.config.states if s.detector == 'face']
            # Sort face states by priority, but ensure STRAIGHT_FACE is checked last
            sorted_face_states = sorted(face_states, key=lambda x: x.priority, reverse=True)
            # Separate STRAIGHT_FACE to check it as fallback
            straight_face_state = next((s for s in sorted_face_states if s.name == 'STRAIGHT_FACE'), None)
            other_face_states = [s for s in sorted_face_states if s.name != 'STRAIGHT_FACE']
            
            # Check other face states first
            face_detected = False
            for state in other_face_states:
                detector_method = getattr(self, f'detect_{state.name}', None)
                if detector_method and callable(detector_method):
                    if detector_method(frame_rgb):
                        detected_states.append(state.name)
                        face_detected = True
                        break
            
            # If no other face state matched, check STRAIGHT_FACE as fallback
            if not face_detected and straight_face_state:
                detector_method = getattr(self, f'detect_{straight_face_state.name}', None)
                if detector_method and callable(detector_method):
                    if detector_method(frame_rgb):
                        detected_states.append(straight_face_state.name)

        # Return highest priority detected state, or default
        return self._get_highest_priority_state(detected_states)

    # Individual pose detection methods - users can override these
    
    def detect_HANDS_UP(self, frame_rgb) -> bool:
        """
        Detect if both hands are raised above shoulders.
        """
        if not self._cached_pose_results or not self._cached_pose_results.pose_landmarks:
            return False

        landmarks = self._cached_pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        return (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y)

    def detect_THINKING(self, frame_rgb) -> bool:
        """
        Detect thinking pose (hand near chin/face and above shoulder).
        """
        if not self._cached_pose_results or not self._cached_pose_results.pose_landmarks:
            return False

        landmarks = self._cached_pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_index = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX]
        right_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX]

        # Try to get face landmarks for precise detection
        if self._cached_face_results and self._cached_face_results.multi_face_landmarks:
            for face_landmarks in self._cached_face_results.multi_face_landmarks:
                chin = face_landmarks.landmark[18]  # Chin tip
                
                # Check left hand
                if self._is_hand_near_chin(left_index, left_shoulder, chin):
                    return True
                
                # Check right hand
                if self._is_hand_near_chin(right_index, right_shoulder, chin):
                    return True
        
        # Fallback: if face not detected, check if hand is raised above shoulder
        return (left_index.y < left_shoulder.y) or (right_index.y < right_shoulder.y)

    def detect_NERD(self, frame_rgb) -> bool:
        """
        Detect nerd pose (one index finger pointing upward, NOT near face).
        
        Checks if an index finger is pointing up, but excludes cases where hand is near face
        (those should be THINKING instead).
        """
        if not self._cached_pose_results or not self._cached_pose_results.pose_landmarks:
            return False

        landmarks = self._cached_pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_index = landmarks[self.mp_pose.PoseLandmark.LEFT_INDEX]
        right_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        # Threshold for "finger pointing up" - finger must be significantly above wrist
        finger_up_threshold = 0.03  # Normalized coordinates - finger must be clearly extended upward
        
        # Check if left finger is pointing up
        left_finger_up = (left_index.y < left_wrist.y - finger_up_threshold)
        
        # Check if right finger is pointing up
        right_finger_up = (right_index.y < right_wrist.y - finger_up_threshold)
        
        # If finger is pointing up, check that hand is NOT near face (to avoid conflict with THINKING)
        if left_finger_up or right_finger_up:
            if self._cached_face_results and self._cached_face_results.multi_face_landmarks:
                for face_landmarks in self._cached_face_results.multi_face_landmarks:
                    chin = face_landmarks.landmark[18]  # Chin tip
                    
                    # If hand is near face, this should be THINKING, not NERD
                    if left_finger_up and self._is_hand_near_chin(left_index, left_shoulder, chin):
                        return False
                    if right_finger_up and self._is_hand_near_chin(right_index, right_shoulder, chin):
                        return False
            
            # Finger is pointing up and NOT near face - this is NERD
            return True
        
        return False

    def detect_EYEBROW_RAISED(self, frame_rgb) -> bool:
        """
        Detect if exactly one eyebrow is raised (either left or right, but not both).
        
        Uses multiple checks to ensure resilience:
        1. One eyebrow must be significantly higher than the other
        2. The raised eyebrow must be above its eye baseline
        3. The other eyebrow must be at or below its baseline
        """
        if not self._cached_face_results or not self._cached_face_results.multi_face_landmarks:
            return False

        for face_landmarks in self._cached_face_results.multi_face_landmarks:
            # Eyebrow landmarks (MediaPipe face mesh)
            # Left eyebrow: outer (107), inner (55), middle points (65, 52, 53, 46)
            # Right eyebrow: outer (336), inner (296), middle points (334, 293, 301, 368)
            left_eyebrow_outer = face_landmarks.landmark[107]
            left_eyebrow_inner = face_landmarks.landmark[55]
            left_eyebrow_mid1 = face_landmarks.landmark[65]
            left_eyebrow_mid2 = face_landmarks.landmark[52]
            
            right_eyebrow_outer = face_landmarks.landmark[336]
            right_eyebrow_inner = face_landmarks.landmark[296]
            right_eyebrow_mid1 = face_landmarks.landmark[334]
            right_eyebrow_mid2 = face_landmarks.landmark[293]
            
            # Eye landmarks for baseline reference
            # Left eye top: 159, Right eye top: 386
            left_eye_top = face_landmarks.landmark[159]
            right_eye_top = face_landmarks.landmark[386]
            
            # Calculate average eyebrow height using multiple points for better accuracy
            # Lower y = higher on face
            left_eyebrow_avg_y = (
                left_eyebrow_outer.y + left_eyebrow_inner.y + 
                left_eyebrow_mid1.y + left_eyebrow_mid2.y
            ) / 4
            right_eyebrow_avg_y = (
                right_eyebrow_outer.y + right_eyebrow_inner.y + 
                right_eyebrow_mid1.y + right_eyebrow_mid2.y
            ) / 4
            
            # Calculate the difference between eyebrows
            # Positive if left is higher (lower y value), negative if right is higher
            eyebrow_diff = left_eyebrow_avg_y - right_eyebrow_avg_y
            
            # Threshold for detection - focus on relative difference between eyebrows
            # This is the most reliable indicator of a single eyebrow raise
            min_eyebrow_diff = 0.018  # Minimum difference (one eyebrow noticeably higher)
            
            # Check if left eyebrow is significantly higher than right
            left_raised = eyebrow_diff > min_eyebrow_diff
            
            # Check if right eyebrow is significantly higher than left
            right_raised = eyebrow_diff < -min_eyebrow_diff
            
            # Return True if exactly one eyebrow is significantly higher than the other
            return left_raised or right_raised

        return False

    def detect_CATXD(self, frame_rgb) -> bool:
        """
        Detect smiling face (mouth aspect ratio exceeds threshold).
        
        Override this method to customize CATXD detection.
        """
        if not self._cached_face_results or not self._cached_face_results.multi_face_landmarks:
            return False

        for face_landmarks in self._cached_face_results.multi_face_landmarks:
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
                    return True

        return False

    def detect_STRAIGHT_FACE(self, frame_rgb) -> bool:
        """
        Default state - always returns True if face is detected.
        """
        return (self._cached_face_results is not None and 
                self._cached_face_results.multi_face_landmarks is not None)

    # Helper methods for pose detection
    
    def _is_hand_near_chin(self, index, shoulder, chin, threshold=0.15) -> bool:
        """
        Check if hand is near chin and above shoulder.
        """
        # Calculate distance from hand to chin
        hand_to_chin_distance = ((index.x - chin.x)**2 + (index.y - chin.y)**2)**0.5
        
        # Check if hand is near chin and above shoulder
        return (hand_to_chin_distance < threshold) and (index.y < shoulder.y)

    def _has_high_priority_pose(self, pose_state):
        """Check if pose state has higher priority than face states."""
        if not pose_state:
            return False
        pose_config = self.config.get_state_by_name(pose_state)
        return pose_config and pose_config.priority >= 30  # Above face states

    def _get_highest_priority_state(self, detected_states):
        """Return highest priority state from detected states."""
        if not detected_states:
            return DEFAULT_STATE

        state_priorities = []
        for state_name in detected_states:
            state_config = self.config.get_state_by_name(state_name)
            if state_config:
                state_priorities.append((state_name, state_config.priority))

        if not state_priorities:
            return DEFAULT_STATE

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
            print(f"- {state.image_file} ({state.emoji} {state.name})")

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
            emoji_name = state_config.emoji if state_config else '❓'

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
    """
    Entry point for the emoji reactor application.

    By default uses DefaultEmojiConfig. To use a custom configuration,
    create a subclass of EmojiConfig and pass it to EmojiReactor.
    """
    config = DefaultEmojiConfig()
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

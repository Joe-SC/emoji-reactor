# Emoji Reactor

A real-time camera-based emoji display application that uses MediaPipe to detect your poses and facial expressions, then displays corresponding emojis in a separate window.

## Features

- **Hand Detection**: Raise both hands above shoulders → displays hands up emoji 🙌
- **One Hand Detection**: Raise one hand → displays one hand emoji ✋
- **Smile Detection**: Detects smiling → displays smiling emoji 😊
- **Default State**: Straight face → displays neutral emoji 😐
- **Real-time Processing**: Live camera feed with instant emoji reactions
- **Config-Driven Architecture**: Easily add new emoji states through configuration

## Requirements

- Python 3.12+
- macOS or Windows with a webcam
- Required Python packages (managed via `uv` or `pip`)

## Installation

### Option A: Using uv (Recommended)

1. **Install uv** (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone or download this project**

3. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

4. **Run the application:**
   ```bash
   uv run emoji_reactor.py
   ```

### Option B: Using pip

1. **Clone or download this project**

2. **Create a virtual environment and install dependencies:**
   ```bash
   # Create and activate a virtual environment
   python3 -m venv emoji_env
   source emoji_env/bin/activate  # On Windows: emoji_env\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Ensure you have the emoji images in the project directory:**
   - `catxd.png` - Smiling face emoji
   - `toletole.png` - Straight face emoji
   - `aircat.png` - Hands up emoji
   - `hmm.png` - One hand up emoji

## Usage

1. **Run the application:**
   ```bash
   # If using uv
   uv run emoji_reactor.py

   # If using pip/venv
   source emoji_env/bin/activate  # On Windows: emoji_env\Scripts\activate
   python emoji_reactor.py

   # Or use helper script
   ./run.sh
   ```

2. **Two windows will open:**
   - **Camera Feed**: Shows your live camera with detection status
   - **Emoji Output**: Displays the corresponding emoji based on your actions

3. **Controls:**
   - Press `q` to quit the application
   - Raise both hands above your shoulders for hands up emoji
   - Raise one hand for one hand emoji
   - Smile for the smiling emoji
   - Keep a straight face for the neutral emoji

## How It Works

The application uses a modular, config-driven architecture with four main components:

1. **EmojiConfig**: Centralized configuration for states, thresholds, and display settings
2. **ImageManager**: Handles loading and managing emoji images
3. **StateDetector**: Uses MediaPipe to detect poses and facial expressions
4. **EmojiReactor**: Main orchestrator that coordinates detection and display

### MediaPipe Detection

The application uses two MediaPipe solutions:

1. **Pose Detection**: Monitors shoulder, wrist, and index finger positions to detect hand gestures
2. **Face Mesh Detection**: Analyzes mouth shape to detect smiling vs. straight face

### Detection Priority (Highest to Lowest)
1. **Hands Up** (priority 40) - Both hands raised above shoulders
2. **One Hand Up** (priority 30) - One index finger above shoulder
3. **Smiling** (priority 20) - Mouth aspect ratio exceeds threshold
4. **Straight Face** (priority 10) - Default state

## Customization

### Adding a New Emoji State

The refactored architecture makes it easy to add new states:

1. **Add state definition** to `EmojiConfig.__init__()` in `emoji_reactor.py`:
   ```python
   {
       'name': 'PEACE_SIGN',
       'image_file': 'peace.png',
       'emoji': '✌️',
       'priority': 35,  # Between ONE_HAND_UP and HANDS_UP
       'detector': 'pose'
   }
   ```

2. **Add detection logic** to `StateDetector._check_pose_state()`:
   ```python
   # Check for peace sign
   if self._is_peace_sign(landmarks):
       return 'PEACE_SIGN'
   ```

3. **Add the image file** (`peace.png`) to the project directory

That's it! The rest (loading, display, priority handling) is automatic.

### Adjusting Smile Sensitivity
Edit the `smile_threshold` value in `EmojiConfig.__init__()`:
- Decrease value (e.g., 0.20) if smiles aren't detected
- Increase value (e.g., 0.30) if false positive smiles occur

### Changing Emoji Images
Replace the image files with your own:
- `catxd.png` - Your smiling emoji
- `toletole.png` - Your neutral emoji
- `aircat.png` - Your hands up emoji
- `hmm.png` - Your one hand emoji

## Troubleshooting

### Camera Issues (macOS)
- If you see "not authorized to capture video", grant Camera access for your terminal/editor:
  - System Settings → Privacy & Security → Camera → enable for Terminal/VS Code/iTerm
- Quit and relaunch the terminal/editor after changing permissions
- Ensure no other app is using the camera
- Try different camera indices by changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Emoji Images Not Loading
- Verify image files are in the same directory as the script
- Check file names match exactly: `catxd.png`, `toletole.png`, `aircat.png`, `hmm.png`
- Ensure image files are not corrupted

### Detection Issues
- Ensure good lighting on your face
- Keep your face clearly visible in the camera
- Adjust `smile_threshold` in `EmojiConfig` if needed
- For hands up detection, make sure your arms are clearly visible

## Technical Details

### Architecture
- **Modular class-based design** with clear separation of concerns
- **Config-driven state management** for easy extensibility
- **Priority-based detection system** for handling multiple simultaneous states

### Technologies
- **OpenCV** - Camera capture and display
- **MediaPipe** - Pose and Face Mesh detection with 33 pose landmarks and 468 face landmarks
- **NumPy** - Numerical computing for landmark calculations

### Code Structure
```
emoji_reactor.py (365 lines)
├── EmojiConfig (68 lines) - Configuration and state definitions
├── ImageManager (35 lines) - Image loading and management
├── StateDetector (118 lines) - Pose and face detection logic
├── EmojiReactor (120 lines) - Main application orchestration
└── main() (18 lines) - Entry point
```

## Dependencies

- `opencv-python>=4.8.0` - Computer vision library
- `mediapipe>=0.10.13` - Pose and Face Mesh detection
- `numpy>=1.24.0` - Numerical computing
- `pillow>=10.0.0` - Image processing

See `pyproject.toml` for project configuration, `requirements.txt` for pip installation, and `uv.lock` for reproducible builds.

## License

MIT License - see LICENSE file for details.

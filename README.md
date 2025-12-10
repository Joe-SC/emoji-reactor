# Emoji Reactor

A real-time camera-based emoji display application that uses MediaPipe to detect your poses and facial expressions, then displays corresponding emojis in a separate window.

## Requirements

- Python 3.12+
- macOS or Windows with a webcam
- Required Python packages (managed via `uv` or `pip`)

## Installation

### Option A: Using uv (preferred)

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

The application uses a config-driven architecture with four main components:

1. **EmojiConfig**: Configuration for states, thresholds, and display settings
2. **ImageManager**: Loads and manages emoji images
3. **StateDetector**: Uses MediaPipe to detect poses and facial expressions
4. **EmojiReactor**: Coordinates detection and display

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

To add a new state:

1. **Update the StateName type** in `emoji_reactor.py`:
   ```python
   StateName = Literal['HANDS_UP', 'ONE_HAND_UP', 'SMILING', 'STRAIGHT_FACE', 'PEACE_SIGN']
   ```

2. **Add state definition** to `DefaultEmojiConfig.__init__()` in `emoji_reactor.py`:
   ```python
   EmojiState(
       name='PEACE_SIGN',
       image_file='peace.png',
       emoji='✌️',
       priority=35,  # Between ONE_HAND_UP and HANDS_UP
       detector='pose'
   )
   ```

3. **Add detection logic** to `StateDetector._check_pose_state()`:
   ```python
   # Check for peace sign
   if self._is_peace_sign(landmarks):
       return 'PEACE_SIGN'
   ```

4. **Add the image file** (`peace.png`) to the project directory

### Creating a Custom Configuration

Extend the base class to create a custom configuration:

```python
class MyEmojiConfig(EmojiConfig):
    """Custom configuration with my own emojis."""

    def __init__(self):
        super().__init__()

        # Customize settings
        self.smile_threshold = 0.30
        self.window_width = 1024
        self.window_height = 768

        # Define your own states
        self.states = [
            EmojiState(
                name='HANDS_UP',
                image_file='my_hands_up.png',
                emoji='🙌',
                priority=40,
                detector='pose'
            ),
            # ... more states
        ]

# Use your custom config
config = MyEmojiConfig()
reactor = EmojiReactor(config)
```

### Adjusting Smile Sensitivity
Edit the `smile_threshold` value in `DefaultEmojiConfig.__init__()` (or create a custom config):
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


## Dependencies
See `pyproject.toml` for project configuration, `requirements.txt` for pip installation, and `uv.lock` for reproducible builds.

## License

MIT License - see LICENSE file for details.

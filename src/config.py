from custom_dataclass import CardConfig, ImageProcessorConfig, PlayerConfig

DEBUG = False  # Enable or disable debugging

# Path to the Tesseract OCR executable
TESSERACT_PATH = "path/to/tesseract.exe"

# List of card suits (club, spade, diamond, heart)
CARD_SUITS = ["club", "spade", "diamond", "heart"]

# List of card colors (black, red)
CARD_COLORS = ["black", "red"]

# Card configuration settings for the game (e.g., dimensions and overlaps)
CARD_CONFIG = CardConfig(
    overlap_open_waste_card_w_ratio=0.2,  # Overlap ratio for open waste cards (width)
    overlap_open_tableau_card_h_ratio=0.32,  # Overlap ratio for open tableau cards (height)
    overlap_unopen_tableau_card_h_ratio=0.1,  # Overlap ratio for unopened tableau cards (height)
    max_tableau_h_ratio=3.64,  # Maximum allowed height for tableau cards
    corner_offset=2,  # Offset for card corner alignment
    corner_w_ratio=0.19,  # Ratio for corner width
    corner_h_ratio=0.27   # Ratio for corner height
)

# Configuration for image processing and OCR
IMAGE_PROCESSOR_CONFIG = ImageProcessorConfig(
    num_candidate_ocr=3,  # Number of candidate OCR results to consider
    upscale_factor=3,  # Upscaling factor for image processing
    contour_area_threshold=20,  # Minimum contour area threshold for detection
    card_suits_dir="media/card_suits"  # Directory containing images of card suits
)

# Configuration for player interaction and movement
PLAYER_CONFIG = PlayerConfig(
    enable_single_tap_to_move=True,  # Enable or disable single tap to move
    action_pause=0.1,  # Pause duration after each action (in seconds)
    animation_pause=0.5,  # Pause duration for animations (in seconds)
    drag_duration=0.5,  # Duration for dragging an item (in seconds)
    move_duration=0.5   # Duration for moving an item (in seconds)
)

# Color ranges for different regions in the image, used for contour detection
CONTOUR_AREA_COLORS = {
    'open_card': ([200, 200, 200], [255, 255, 255]),  # Color range for opened card area (grayscale)
    'unopen_card': ([150, 0, 0], [255, 100, 100]),    # Color range for unopened card area (red tones)
    'background': ([0, 100, 0], [100, 255, 100]),     # Color range for background area (green tones)
}

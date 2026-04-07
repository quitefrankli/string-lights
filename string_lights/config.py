import cv2
from enum import Enum

# Board geometry
BOARD_COLS  = 5
BOARD_ROWS  = 5
SQUARE_SIZE = 0.018   # metres
MARKER_SIZE = 0.013   # metres
CHARUCO_BOARD_WIDTH = SQUARE_SIZE * BOARD_COLS
ARUCO_DICT  = cv2.aruco.DICT_6X6_50

# Camera model
FOCAL_PX_AT_1920 = 1800.0

# Pose validation
MAX_ROTATION_JUMP    = 0.3    # radians per frame
MAX_TRANSLATION_JUMP = 0.05   # metres per frame


class PoseResolution(Enum):
    OMIT        = "omit"         # invalid frames output (None, None)
    HOLD        = "hold"         # repeat last valid pose
    INTERPOLATE = "interpolate"  # linearly interpolate between surrounding valid poses

POSE_RESOLUTION = PoseResolution.HOLD

# Hand masking (SAM)
GD_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL_ID = "facebook/sam-vit-base"
MASK_PROMPT = "hands"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
MASK_FRAME_SKIP = 1  # reuse mask for N frames

# String rendering
NUM_STRINGS = 6
STRING_COLOR = (200, 220, 255)  # warm white (BGR)
STRING_CORE_COLOR = (255, 255, 255)
STRING_ALPHA = 0.8
FADE_DURATION = 0.5  # seconds

# generate with `tuner`
STRING_SPACING_FACTOR = 0.4
STRING0_OFFSET_X_FACTOR = -1.72
STRING0_OFFSET_Y_FACTOR = 0.65
STRING_LENGTH_FACTOR = 6.63
STRING_CONVERGENCE_FACTOR = 1.55
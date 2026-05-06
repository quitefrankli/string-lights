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

# Pose smoothing (One-Euro filter, applied after pose resolution, before render)
SMOOTH_POSES = True
T_MIN_CUTOFF = 1.0   # Hz; lower = more smoothing of tvec when still
T_BETA = 10         # responsiveness to fast translation (units: per m/s)
R_MIN_CUTOFF = 1.0   # Hz; lower = more smoothing of rotation when still
R_BETA = 30         # responsiveness to fast rotation (units: per quat-speed)

# Hand masking
GD_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# SAM2_MODEL_ID = "facebook/sam2-hiera-small"
# SAM2_MODEL_ID = "facebook/sam2-hiera-base-plus"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
MASK_PROMPT = "hands"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
MASK_THRESHOLD = 0.0   # SAM2 logit threshold; higher = tighter mask boundary
MASK_DILATE_PX = 3    # dilation after masking to fill gaps
MASK_FRAME_SKIP = 1

# String rendering
NUM_STRINGS = 6
STRING_COLOR = (200, 220, 255)  # warm white (BGR)
STRING_CORE_COLOR = (255, 255, 255)
STRING_ALPHA = 0.8
FADE_DURATION = 0.5  # seconds

# Four corner points of the string array, in board coordinates (metres).
# String i endpoint = lerp(top, bot, i / (NUM_STRINGS - 1)) at each end.
# Generate with `tuner` (drag corners on image).
STRING_BRIDGE_TOP = (0.162169, -0.010443, 0.000000)
STRING_BRIDGE_BOT = (0.154770, -0.065316, 0.000000)
STRING_NUT_TOP = (0.757137, -0.014816, 0.000000)
STRING_NUT_BOT = (0.756080, -0.050829, 0.000000)
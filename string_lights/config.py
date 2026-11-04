# Hand masking
MASK_MODEL_ID = "FudanCVL/SAM2Matting"
MASK_CHECKPOINT = "checkpoints/SAM2Matting-SAM2.1Base+.pt"
MASK_CONFIG = "configs/sam2matting-sam2.1base+.yaml"
MASK_ENCODING = "alpha_u8"

# String rendering
NUM_STRINGS = 6
STRING_COLOR = (200, 220, 255)  # warm white (BGR)
STRING_CORE_COLOR = (255, 255, 255)
STRING_ALPHA = 0.8
FADE_DURATION = 0.5  # seconds

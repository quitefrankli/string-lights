import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)

from .config import GD_MODEL_ID, SAM_MODEL_ID


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(device: str):
    print(f"  loading GroundingDINO ({GD_MODEL_ID})...")
    gd_processor = AutoProcessor.from_pretrained(GD_MODEL_ID)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL_ID).to(device)

    print(f"  loading SAM ({SAM_MODEL_ID})...")
    sam_processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    sam_model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)

    gd_model.eval()
    sam_model.eval()
    return gd_processor, gd_model, sam_processor, sam_model


def get_mask(
    frame_bgr: np.ndarray,
    prompt: str,
    gd_processor,
    gd_model,
    sam_processor,
    sam_model,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    gd_prompt = prompt if prompt.endswith(".") else prompt + "."
    inputs = gd_processor(images=pil_image, text=gd_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)

    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"]
    if boxes.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    input_boxes = boxes.cpu().numpy().tolist()
    sam_inputs = sam_processor(
        images=pil_image,
        input_boxes=[input_boxes],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]

    iou_scores = sam_outputs.iou_scores[0].cpu()
    best_indices = iou_scores.argmax(dim=1)

    combined = np.zeros((h, w), dtype=np.uint8)
    for i in range(masks.shape[0]):
        best = best_indices[i].item()
        m = masks[i, best].numpy().astype(np.uint8)
        combined = np.maximum(combined, m)

    return combined

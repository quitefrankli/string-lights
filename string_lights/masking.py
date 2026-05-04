import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Model,
    Sam2Processor,
)

from .config import GD_MODEL_ID, SAM2_MODEL_ID, MASK_THRESHOLD, MASK_DILATE_PX


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

    print(f"  loading SAM2 ({SAM2_MODEL_ID})...")
    sam_processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
    sam_model = Sam2Model.from_pretrained(SAM2_MODEL_ID).to(device)

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
    debug_writer: cv2.VideoWriter | None = None,
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

    # pred_masks: [batch, num_boxes, num_candidates, mask_h, mask_w]
    pred_masks = sam_outputs.pred_masks[0].cpu()
    iou_scores = sam_outputs.iou_scores[0].cpu()
    best_indices = iou_scores.argmax(dim=1)

    combined = np.zeros((h, w), dtype=np.uint8)
    for i in range(pred_masks.shape[0]):
        best = best_indices[i].item()
        m = pred_masks[i, best:best+1].unsqueeze(0).float()
        m_up = torch.nn.functional.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        combined = np.maximum(combined, (m_up[0, 0] > MASK_THRESHOLD).numpy().astype(np.uint8))

    if MASK_DILATE_PX > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_PX * 2 + 1,) * 2)
        combined = cv2.dilate(combined, kernel)

    if debug_writer is not None:
        vis = frame_bgr.copy()
        overlay = vis.copy()
        overlay[combined.astype(bool)] = (0, 0, 200)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        scores = results["scores"].cpu().numpy()
        for box, score in zip(input_boxes, scores):
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 2)
            cv2.putText(vis, f"{score:.2f}", (x0, y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        debug_writer.write(vis)

    return combined

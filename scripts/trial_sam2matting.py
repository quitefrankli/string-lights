"""Run a SAM2Matting Base+ trial against an existing binary hand-mask cache."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--source", type=Path, required=True, help="Official SAM2Matting checkout")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from sam2.build_sam import build_sam2matting_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("This trial requires CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = hf_hub_download("FudanCVL/SAM2Matting", "checkpoints/SAM2Matting-SAM2.1Base+.pt")
    baseline_file = np.load(args.baseline)
    baseline = baseline_file["masks"]
    cap = cv2.VideoCapture(str(args.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if baseline.shape != (total, height, width):
        raise ValueError("Baseline must contain one full-resolution mask for every frame")
    seed = baseline[0].astype(np.uint8)
    cv2.imwrite(str(args.output / "seed.png"), seed * 255)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    predictor = build_sam2matting_video_predictor(
        "configs/sam2matting-sam2.1base+.yaml", checkpoint, device="cuda", hydra_overrides_extra=["++model.fill_hole_area=0"]
    )
    # Upstream appends fill_hole_area=8 after user overrides; disable the optional CUDA extension.
    predictor.fill_hole_area = 0
    alpha_cache = np.lib.format.open_memmap(
        args.output / "alpha.npy", mode="w+", dtype=np.uint8, shape=(total, height, width)
    )
    # JPEG loading is the upstream reference path and avoids an extra video decoder dependency.
    with tempfile.TemporaryDirectory(prefix="sam2matting-frames-") as frame_dir:
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not decode frame {i}")
            if not cv2.imwrite(str(Path(frame_dir) / f"{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                raise RuntimeError("Could not write reference frames")
        cap.release()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=frame_dir, offload_video_to_cpu=False)
            # Follow the upstream inference script's initial-mask preprocessing.
            mask = torch.from_numpy(seed.astype(np.float32) * 20 - 10)[None, None]
            mask = F.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False)
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask.cuda())
            count = 0
            inference_start = time.perf_counter()
            for index, _, _, alpha, _ in predictor.propagate_in_video(state):
                if index != count:
                    raise RuntimeError(f"Unexpected output frame {index}, expected {count}")
                matte = alpha.detach().float().cpu().squeeze().numpy().clip(0, 1)
                if matte.shape != (height, width):
                    raise ValueError(f"Unexpected alpha shape {matte.shape}")
                alpha_cache[index] = np.rint(matte * 255).astype(np.uint8)
                count += 1
                if count % 30 == 0 or count == total:
                    print(f"Matting: {count}/{total}", flush=True)
            inference_seconds = time.perf_counter() - inference_start
    alpha_cache.flush()
    if count != total:
        raise RuntimeError(f"Incomplete output: {count}/{total}")
    stats = dict(model="SAM2Matting-SAM2.1Base+", width=width, height=height, fps=fps,
                 frames=count, inference_seconds=inference_seconds,
                 total_seconds=time.perf_counter() - start,
                 peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 1024**2,
                 source_commit=subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
                 seed_source=str(args.baseline), alpha_encoding="uint8 opacity 0..255")
    (args.output / "metadata.json").write_text(json.dumps(stats, indent=2) + "\n")
    del predictor, state
    torch.cuda.empty_cache()
    cap = cv2.VideoCapture(str(args.video))
    preview = args.output / "comparison.mp4"
    encoder = subprocess.Popen([
        "ffmpeg", "-nostdin", "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "1548x540", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preview),
    ], stdin=subprocess.PIPE)
    try:
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not decode preview frame {i}")
            panels = []
            for label, opacity in (("SAM 2.1 + GroundingDINO", baseline[i]),
                                   ("SAM2Matting Base+", alpha_cache[i].astype(np.float32) / 255)):
                weight = opacity[..., None] * 0.55
                overlay = (frame * (1 - weight) + np.array([30, 30, 255]) * weight).astype(np.uint8)
                panel = cv2.resize(overlay, (774, 540))
                cv2.rectangle(panel, (0, 0), (774, 40), (0, 0, 0), -1)
                cv2.putText(panel, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(panel)
            combined = np.concatenate(panels, axis=1)
            encoder.stdin.write(combined.tobytes())
            if i in (0, 60, 120, total - 1):
                cv2.imwrite(str(args.output / f"comparison_{i:03d}.png"), combined)
    finally:
        cap.release()
        encoder.stdin.close()
        encoder.wait()
    if encoder.returncode:
        raise RuntimeError("Preview encoding failed")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"Preview: {preview}", flush=True)


if __name__ == "__main__":
    main()

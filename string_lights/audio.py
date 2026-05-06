import random
from pathlib import Path

import numpy as np

AUDIO_FPS = 22050 / 512  # ~43.06 fps
MIN_NOTE_FRAMES = 2  # ignore notes shorter than this (audio frames)


def get_random_strings(total_frames: int, video_fps: float, seed: int | None = None) -> list[list[int]]:
    """Generate random string highlights — strums every ~0.4-0.8s with 1-3 strings each."""
    rng = random.Random(seed)
    strings: list[list[int]] = [[] for _ in range(total_frames)]
    i = 0
    while i < total_frames:
        n = rng.randint(1, 3)
        picks = rng.sample(range(6), n)
        strings[i] = picks
        gap = rng.randint(int(video_fps * 0.4), max(int(video_fps * 0.8), int(video_fps * 0.4) + 1))
        i += gap
    return strings


def _clean_tab(tab: np.ndarray) -> np.ndarray:
    """Per-string: drop short notes, collapse sustained notes to onset-only."""
    n, n_strings = tab.shape
    out = np.zeros_like(tab)
    for s in range(n_strings):
        col = tab[:, s]
        i = 0
        while i < n:
            if col[i] == 0:
                i += 1
                continue
            j = i
            while j < n and col[j] == col[i]:
                j += 1
            if j - i >= MIN_NOTE_FRAMES:
                out[i, s] = col[i]
            i = j
    return out


def get_strings_to_highlight(input_path: str, total_frames: int, video_fps: float) -> list[list[int]]:
    npy_path = Path(input_path).with_suffix(".npy")
    if not npy_path.exists():
        raise FileNotFoundError(f"Tab data not found: {npy_path}")

    tab = np.load(npy_path)  # (N, 6) int — columns are E2 A2 D3 G3 B3 E4
    n_audio = len(tab)
    print(f"Loaded tab data from {npy_path}, {n_audio} audio frames, {total_frames} video frames")

    # tab = _clean_tab(tab)

    strings: list[list[int]] = []
    for vi in range(total_frames):
        ai = int(vi * AUDIO_FPS / video_fps)
        if ai >= n_audio:
            strings.append([])
            continue
        row = tab[ai]
        strings.append([5 - s for s in range(6) if row[s] > 0])
    return strings

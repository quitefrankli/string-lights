import random

from .strings import NUM_STRINGS


def get_strings_to_highlight(total_frames: int, fps: float) -> list[list[int]]:
    """Return per-frame list of active string indices. Sparse random dummy output for now."""
    rng = random.Random(42)
    strings: list[list[int]] = [[] for _ in range(total_frames)]
    frame = 0
    while frame < total_frames:
        count = rng.randint(1, 3)
        indices = rng.sample(range(NUM_STRINGS), min(count, NUM_STRINGS))
        strings[frame] = indices
        frame += rng.randint(int(fps * 0.5), int(fps * 2))
    return strings

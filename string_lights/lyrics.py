import json
import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

LYRICS_DIR = Path("data/lyrics")
_FONT_CACHE: dict[int, any] = {}
_FONT_PATHS = [
    "/usr/share/fonts/truetype/quicksand/Quicksand-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def lyrics_path(stem: str) -> Path:
    return LYRICS_DIR / f"{stem}.json"


def load_lyrics(stem: str) -> list[dict]:
    p = lyrics_path(stem)
    return json.loads(p.read_text()) if p.exists() else []


def save_lyrics(stem: str, lyrics: list[dict]) -> None:
    p = lyrics_path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lyrics, indent=2))


def _font(size: int):
    if size not in _FONT_CACHE:
        f = None
        for path in _FONT_PATHS:
            try:
                f = ImageFont.truetype(path, size)
                break
            except (IOError, OSError):
                pass
        _FONT_CACHE[size] = f or ImageFont.load_default()
    return _FONT_CACHE[size]


def draw_lyrics_frame(frame: np.ndarray, frame_idx: int, fps: float, lyrics: list[dict]) -> None:
    if not lyrics or not _PIL:
        return
    t = frame_idx / fps
    h, w = frame.shape[:2]
    active = [l for l in lyrics if l.get("start", 0) <= t < l.get("end", 0) and l.get("text")]
    if not active:
        return

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    for li, lyric in enumerate(active):
        hold = max(1, round(fps / 10))  # ~100 ms per jitter step
        rng = np.random.default_rng((frame_idx // hold) * 9973 + li * 997)
        size = int(lyric.get("size", 72))
        color = tuple(lyric.get("color", [255, 255, 255]))
        px = int(lyric.get("x", 0.5) * w)
        py = int(lyric.get("y", 0.85) * h)
        jitter = float(lyric.get("jitter", 1.0))
        pil = _draw_scribble(pil, lyric["text"], px, py, color, size, _font(size), rng, jitter)

    np.copyto(frame, cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR))


def _draw_scribble(
    pil: "Image.Image",
    text: str,
    cx: int,
    cy: int,
    color: tuple,
    size: int,
    font,
    rng: np.random.Generator,
    jitter: float = 1.0,
) -> "Image.Image":
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stroke_w = max(2, size // 18)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x0 = cx - tw // 2
    y0 = cy - th // 2

    jx_range = round(jitter)
    jy_range = round(jitter * 1.5)

    char_x = x0
    for char in text:
        jx = int(rng.integers(-jx_range, jx_range + 1)) if jx_range > 0 else 0
        jy = int(rng.integers(-jy_range, jy_range + 1)) if jy_range > 0 else 0
        cbbox = draw.textbbox((0, 0), char, font=font)
        advance = cbbox[2] - cbbox[0]
        draw.text(
            (char_x + jx, y0 + jy), char, font=font,
            fill=(*color, 240),
            stroke_width=stroke_w,
            stroke_fill=(0, 0, 0, 210),
        )
        char_x += max(advance, size // 3)  # guard against 0-advance (space)

    return Image.alpha_composite(pil, overlay)

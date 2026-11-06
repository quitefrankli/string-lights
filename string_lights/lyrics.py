import json
from functools import lru_cache
import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    _PIL = True
except ImportError:
    _PIL = False

_FONT_CACHE: dict[tuple[int, bool], any] = {}
_JAPANESE_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
_FONT_PATHS = [
    "/usr/share/fonts/truetype/quicksand/Quicksand-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def lyrics_path(stem: str) -> Path:
    from .paths import artifact_path
    return artifact_path(stem, "lyrics")


def load_lyrics(stem: str) -> list[dict]:
    p = lyrics_path(stem)
    return json.loads(p.read_text()) if p.exists() else []


def save_lyrics(stem: str, lyrics: list[dict]) -> None:
    p = lyrics_path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lyrics, indent=2))


def _font(size: int, text: str = ""):
    japanese = any("\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff" for char in text)
    key = (size, japanese)
    if key not in _FONT_CACHE:
        f = None
        paths = _JAPANESE_FONT_PATHS + _FONT_PATHS if japanese else _FONT_PATHS
        for path in paths:
            try:
                f = ImageFont.truetype(path, size)
                break
            except (IOError, OSError):
                pass
        _FONT_CACHE[key] = f or ImageFont.load_default()
    return _FONT_CACHE[key]


# Bound the cache: exports may contain many different lines and styles.


@lru_cache(maxsize=32)
def _lyric_overlay(w: int, h: int, style: tuple):
    (text, x, y, size, color, opacity, padding, softness,
     halo_size, halo_intensity) = style
    font = _font(size, text)
    outline = max(1, round(size / 36))
    intensity = round(255 * halo_intensity)
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bounds = draw.textbbox((x, y), text, font=font, anchor="mm")
    text_bounds = draw.textbbox((x, y), text, font=font, anchor="mm", stroke_width=outline)
    shadow_offset = max(1, size / 24)
    # Pillow's Gaussian approximation has finite support. Leave extra transparent
    # pixels around it so cropping does not introduce a visible boundary.
    blur_margin = int(np.ceil(4 * halo_size)) + 4 if intensity else 0
    margin = max(padding + softness if opacity else 0, blur_margin)
    left = max(0, int(np.floor(min(bounds[0], text_bounds[0]) - margin)))
    top = max(0, int(np.floor(min(bounds[1], text_bounds[1]) - margin)))
    right = min(w, int(np.ceil(max(bounds[2], text_bounds[2]) + margin)))
    bottom = min(h, int(np.ceil(max(bounds[3], text_bounds[3]) + margin + shadow_offset)))
    if right <= left or bottom <= top:
        return None
    shape = (right - left, bottom - top)
    position = (x - left, y - top)
    backdrop = None
    background_alpha = None
    if opacity > 0:
        def ramp(coordinates, lo, hi):
            if softness == 0:
                return np.ones_like(coordinates, dtype=float)
            u = np.clip(np.minimum(coordinates - lo, hi - coordinates) / softness, 0, 1)
            return u * u * (3 - 2 * u)

        horizontal = ramp(np.arange(left, right) + .5, bounds[0] - padding - softness, bounds[2] + padding + softness)
        vertical = ramp(np.arange(top, bottom) + .5, bounds[1] - padding - softness, bounds[3] + padding + softness)
        # Retain unrounded alpha to preserve the original fade rounding.
        background_alpha = 255 * opacity * vertical[:, None] * horizontal[None, :]
        # The previous renderer clipped the background to this rectangle.
        bg_left, bg_top = np.floor(np.array(bounds[:2]) - padding - softness).astype(int)
        bg_right, bg_bottom = np.ceil(np.array(bounds[2:]) + padding + softness).astype(int)
        background_alpha[:, np.arange(left, right) < bg_left] = 0
        background_alpha[:, np.arange(left, right) >= bg_right] = 0
        background_alpha[np.arange(top, bottom) < bg_top, :] = 0
        background_alpha[np.arange(top, bottom) >= bg_bottom, :] = 0
        backdrop = Image.new("RGBA", shape)
        backdrop.putalpha(Image.fromarray(np.rint(background_alpha).astype(np.uint8)))
    layer = Image.new("RGBA", shape)
    if intensity:
        ImageDraw.Draw(layer).text(
            (position[0], position[1] + shadow_offset), text, font=font,
            anchor="mm", fill=(0, 0, 0, intensity), stroke_width=outline,
            stroke_fill=(0, 0, 0, intensity),
        )
        layer = layer.filter(ImageFilter.GaussianBlur(halo_size))
    ImageDraw.Draw(layer).text(
        position, text, font=font, anchor="mm", fill=(*color, 255),
        stroke_width=outline, stroke_fill=(0, 0, 0, intensity),
    )
    return (left, top, right, bottom), backdrop, background_alpha, layer


def draw_lyrics_frame(frame: np.ndarray, frame_idx: int, fps: float, lyrics: list[dict]) -> None:
    t = frame_idx / fps
    h, w = frame.shape[:2]
    for lyric in lyrics:
        if not (lyric["start"] <= t < lyric["end"] and lyric["text"]):
            continue
        fade = min(lyric.get("fade", .25), (lyric["end"] - lyric["start"]) / 2)
        alpha = min(1, (t - lyric["start"]) / fade, (lyric["end"] - t) / fade) if fade else 1
        if alpha <= 0:
            continue
        style = (
            lyric["text"], lyric.get("x", .5) * w, lyric.get("y", .85) * h,
            int(lyric.get("size", 48)), tuple(lyric.get("color", [255, 255, 255])),
            lyric.get("background_opacity", 0), lyric.get("background_padding", 16),
            lyric.get("background_softness", 24), lyric.get("halo_size", 4),
            lyric.get("halo_intensity", .75),
        )
        overlay = _lyric_overlay(w, h, style)
        if overlay is None:
            continue
        (left, top, right, bottom), backdrop, background_alpha, layer = overlay
        region = frame[top:bottom, left:right]
        pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)).convert("RGBA")
        if backdrop is not None:
            if alpha < 1:
                backdrop = backdrop.copy()
                backdrop.putalpha(Image.fromarray(np.rint(background_alpha * alpha).astype(np.uint8)))
            pil.alpha_composite(backdrop)
        if alpha < 1:
            layer = layer.copy()
            layer.putalpha(layer.getchannel("A").point(lambda a: round(a * alpha)))
        pil.alpha_composite(layer)
        np.copyto(region, cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR))

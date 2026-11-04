import json
import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    _PIL = True
except ImportError:
    _PIL = False

_FONT_CACHE: dict[int, any] = {}
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
    t = frame_idx / fps
    h, w = frame.shape[:2]
    active = [l for l in lyrics if l["start"] <= t < l["end"] and l["text"]]
    if not active:
        return
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    for lyric in active:
        fade = min(lyric.get("fade", 0.25), (lyric["end"] - lyric["start"]) / 2)
        alpha = min(1, (t - lyric["start"]) / fade, (lyric["end"] - t) / fade) if fade else 1
        layer = Image.new("RGBA", pil.size)
        draw = ImageDraw.Draw(layer)
        x, y = lyric.get("x", .5) * w, lyric.get("y", .85) * h
        font = _font(int(lyric.get("size", 48)))
        opacity = lyric.get("background_opacity", 0) * alpha
        if opacity > 0:
            bounds = draw.textbbox((x, y), lyric["text"], font=font, anchor="mm")
            padding = lyric.get("background_padding", 16)
            softness = lyric.get("background_softness", 24)
            left, top = np.floor(np.array(bounds[:2]) - padding - softness).astype(int)
            right, bottom = np.ceil(np.array(bounds[2:]) + padding + softness).astype(int)
            left, top = max(0, left), max(0, top)
            right, bottom = min(w, right), min(h, bottom)
            if right > left and bottom > top:
                # A smoothstep ramp on each edge gives a flat dark center and
                # continuously fading edges, including at the corners.
                def ramp(coordinates, lo, hi):
                    if softness == 0:
                        return np.ones_like(coordinates, dtype=float)
                    u = np.clip(np.minimum(coordinates - lo, hi - coordinates) / softness, 0, 1)
                    return u * u * (3 - 2 * u)

                horizontal = ramp(np.arange(left, right) + .5, bounds[0] - padding - softness, bounds[2] + padding + softness)
                vertical = ramp(np.arange(top, bottom) + .5, bounds[1] - padding - softness, bounds[3] + padding + softness)
                backdrop = Image.new("RGBA", (right - left, bottom - top))
                backdrop.putalpha(Image.fromarray(np.rint(255 * opacity * vertical[:, None] * horizontal[None, :]).astype(np.uint8)))
                pil.alpha_composite(backdrop, (left, top))
        size = int(lyric.get("size", 48))
        outline = max(1, round(size / 36))
        intensity = round(255 * lyric.get("halo_intensity", .75))
        shadow = Image.new("RGBA", pil.size)
        ImageDraw.Draw(shadow).text(
            (x, y + max(1, size / 24)), lyric["text"], font=font,
            anchor="mm", fill=(0, 0, 0, intensity), stroke_width=outline,
            stroke_fill=(0, 0, 0, intensity),
        )
        layer = shadow.filter(ImageFilter.GaussianBlur(lyric.get("halo_size", 4)))
        draw = ImageDraw.Draw(layer)
        draw.text((x, y), lyric["text"], font=font, anchor="mm",
                  fill=(*lyric.get("color", [255,255,255]),255),
                  stroke_width=outline, stroke_fill=(0, 0, 0, intensity))
        layer.putalpha(layer.getchannel("A").point(lambda a: round(a * alpha)))
        pil = Image.alpha_composite(pil, layer)
    np.copyto(frame, cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR))

"""
degradations.py
----------------
Realistic phone-camera image degradations at graded severities.

Each function takes a PIL.Image (RGB) and a severity in {1,2,3} (low/mid/high)
and returns a degraded PIL.Image. These approximate real-world artifacts that
occur when people take and share photos on phones: compression, motion blur,
poor lighting, glare, and tilt.

Design notes for the paper:
- We deliberately use REALISTIC phone artifacts, not only ImageNet-C synthetic
  noise. This is part of our novelty framing.
- Severities are graded so we can plot accuracy/ECE vs. severity curves.
- All transforms are deterministic given a seed, for reproducibility.
"""

from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def jpeg_compression(img: Image.Image, severity: int = 2) -> Image.Image:
    """Real upload/messaging artifact. Lower quality = more severe."""
    img = _to_rgb(img)
    quality = {1: 30, 2: 15, 3: 7}[severity]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def motion_blur(img: Image.Image, severity: int = 2) -> Image.Image:
    """Handheld shake. Approximated with a directional box blur."""
    img = _to_rgb(img)
    radius = {1: 2, 2: 4, 3: 7}[severity]
    # Simple horizontal motion kernel via repeated shifts + average
    arr = np.asarray(img).astype(np.float32)
    k = radius * 2 + 1
    acc = np.zeros_like(arr)
    for shift in range(-radius, radius + 1):
        acc += np.roll(arr, shift, axis=1)
    acc /= k
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))


def low_light(img: Image.Image, severity: int = 2) -> Image.Image:
    """Underexposure. Reduce brightness and add mild sensor noise."""
    img = _to_rgb(img)
    factor = {1: 0.5, 2: 0.3, 3: 0.15}[severity]
    img = ImageEnhance.Brightness(img).enhance(factor)
    arr = np.asarray(img).astype(np.float32)
    noise_sigma = {1: 5, 2: 10, 3: 18}[severity]
    rng = np.random.default_rng(0)
    arr += rng.normal(0, noise_sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def glare(img: Image.Image, severity: int = 2) -> Image.Image:
    """Overexposure / washed-out from light source."""
    img = _to_rgb(img)
    factor = {1: 1.6, 2: 2.2, 3: 3.0}[severity]
    return ImageEnhance.Brightness(img).enhance(factor)


def rotation(img: Image.Image, severity: int = 2) -> Image.Image:
    """Slight camera tilt. Fills corners with edge padding."""
    img = _to_rgb(img)
    angle = {1: 5, 2: 12, 3: 20}[severity]
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


def resample(img: Image.Image, severity: int = 2) -> Image.Image:
    """Downscale then upscale — loss from resizing on share/upload."""
    img = _to_rgb(img)
    w, h = img.size
    scale = {1: 0.5, 2: 0.3, 3: 0.15}[severity]
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# Registry: name -> function
DEGRADATIONS = {
    "jpeg": jpeg_compression,
    "motion_blur": motion_blur,
    "low_light": low_light,
    "glare": glare,
    "rotation": rotation,
    "resample": resample,
}

SEVERITIES = [1, 2, 3]


def apply_degradation(img: Image.Image, name: str, severity: int) -> Image.Image:
    """Apply a named degradation at a given severity. 'clean' is a no-op."""
    if name == "clean":
        return _to_rgb(img)
    if name not in DEGRADATIONS:
        raise ValueError(f"Unknown degradation '{name}'. "
                         f"Options: {['clean'] + list(DEGRADATIONS)}")
    return DEGRADATIONS[name](img, severity)


if __name__ == "__main__":
    # Quick self-test with a synthetic gradient image (no external deps/files).
    base = Image.fromarray(
        (np.tile(np.linspace(0, 255, 128), (128, 1))).astype(np.uint8)
    ).convert("RGB")
    print("Self-testing degradations on a 128x128 synthetic image...")
    for name in ["clean"] + list(DEGRADATIONS):
        for sev in ([0] if name == "clean" else SEVERITIES):
            out = apply_degradation(base, name, max(sev, 1))
            assert out.size == base.size, f"{name} changed size!"
            print(f"  ok: {name:12s} severity={sev} -> {out.size} {out.mode}")
    print("All degradation self-tests passed.")

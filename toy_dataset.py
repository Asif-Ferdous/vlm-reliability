"""
toy_dataset.py
---------------
Generates a tiny synthetic "shape classification" dataset so the ENTIRE
pipeline runs end-to-end with zero downloads. Each item is an image of a
colored shape plus a multiple-choice question.

This exists ONLY to validate the pipeline logic. For the real paper you will
swap this for a genuine VQA / classification dataset (see load_real_dataset stub).

Convention used across the codebase:
    options[0] is ALWAYS the correct answer (the runner shuffles for display but
    tracks the gold separately). Keeps mock evaluation simple and unambiguous.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from PIL import Image, ImageDraw


SHAPES = ["circle", "square", "triangle"]
COLORS = {"red": (220, 40, 40), "green": (40, 180, 40), "blue": (40, 80, 220)}


@dataclass
class Item:
    image: Image.Image
    question: str
    options: list[str]   # options[0] = correct answer
    gold: str


def _draw_shape(shape: str, color, size=224) -> Image.Image:
    img = Image.new("RGB", (size, size), (245, 245, 245))
    d = ImageDraw.Draw(img)
    m = size // 4
    if shape == "circle":
        d.ellipse([m, m, size - m, size - m], fill=color)
    elif shape == "square":
        d.rectangle([m, m, size - m, size - m], fill=color)
    elif shape == "triangle":
        d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=color)
    return img


def make_dataset(n: int = 60, seed: int = 0) -> list[Item]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        shape = rng.choice(SHAPES)
        cname = rng.choice(list(COLORS))
        img = _draw_shape(shape, COLORS[cname])
        # question about the shape
        options = [shape] + [s for s in SHAPES if s != shape]
        items.append(Item(image=img,
                          question="What shape is shown in the image?",
                          options=options,
                          gold=shape))
    return items


def load_real_dataset(name: str, split: str = "validation", n: int = 500):
    """
    STUB for Week 1-2: swap in a real dataset here.

    Example (fill in when ready):
        from datasets import load_dataset
        ds = load_dataset(name, split=split).select(range(n))
        # map each row to an Item(image=..., question=..., options=..., gold=...)

    Good candidates:
      - A subset of a standard VQA benchmark, or
      - An image-classification set posed as multiple-choice.
    """
    raise NotImplementedError(
        "load_real_dataset is a stub. Use make_dataset() for pipeline validation; "
        "wire in a real dataset in Week 1-2."
    )


if __name__ == "__main__":
    ds = make_dataset(n=6, seed=0)
    for i, it in enumerate(ds):
        print(f"item {i}: gold={it.gold:8s} options={it.options} "
              f"img={it.image.size}")
    print(f"\nGenerated {len(ds)} toy items successfully.")

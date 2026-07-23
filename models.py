"""
models.py
----------
Unified interface for VLMs, with two backends:

  1. MockVLM     -- no GPU, no downloads. Simulates a plausibly-overconfident
                    VLM so you can validate the ENTIRE pipeline end-to-end
                    before touching real hardware. Great for dev + CI.

  2. HFVLMBackend -- real Hugging Face VLM (Qwen2-VL, InternVL, LLaVA, etc.).
                    Stub with clear TODOs; fill in when you have a GPU.

Every backend implements:
    answer(image, question, options) -> (answer_str, confidence_float_0_to_1)

The confidence is the model's VERBALIZED confidence, parsed from its text output
and normalized to [0,1].
"""

from __future__ import annotations
import re
import random
from dataclasses import dataclass
from typing import Optional
from PIL import Image


# ----------------------------- Prompting ---------------------------------

CONFIDENCE_PROMPT = (
    "Look at the image and answer the question.\n"
    "Question: {question}\n"
    "Options: {options}\n\n"
    "Respond in EXACTLY this format:\n"
    "Answer: <one option>\n"
    "Confidence: <an integer from 0 to 100>\n"
)

# RQ3 variant: ask the model to consider image quality first (training-free fix #2)
CONFIDENCE_PROMPT_QUALITY_AWARE = (
    "Look at the image and answer the question.\n"
    "First, briefly consider whether the image is clear or degraded "
    "(blurry, dark, compressed, etc.), because that should affect how "
    "confident you are.\n"
    "Question: {question}\n"
    "Options: {options}\n\n"
    "Respond in EXACTLY this format:\n"
    "Reasoning: <one short sentence on image quality>\n"
    "Answer: <one option>\n"
    "Confidence: <an integer from 0 to 100>\n"
)


def parse_answer_and_confidence(text: str) -> tuple[str, Optional[float]]:
    """Extract answer string and confidence in [0,1] from model output text."""
    answer = None
    conf = None
    m = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    if m:
        answer = m.group(1).strip().splitlines()[0].strip()
    m = re.search(r"Confidence:\s*([0-9]{1,3}(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        conf = max(0.0, min(1.0, val / 100.0))
    return (answer or "").strip(), conf


# ----------------------------- Backends ----------------------------------

@dataclass
class MockVLM:
    """
    Simulates a VLM whose accuracy degrades with image corruption but whose
    stated confidence stays stubbornly high (i.e. overconfident) -- the exact
    phenomenon the paper investigates. Deterministic given seed.

    The mock reads an optional 'severity' hint attached to the image via
    image.info['severity'] (0 = clean). This lets the pipeline demonstrate the
    accuracy-drops-but-confidence-stays-high pattern without a real model.
    """
    seed: int = 0
    base_accuracy: float = 0.82
    overconfidence: float = 0.90  # mean stated confidence, roughly fixed

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def answer(self, image: Image.Image, question: str, options: list[str]):
        severity = int(image.info.get("severity", 0)) if hasattr(image, "info") else 0
        # accuracy falls as severity rises
        acc = max(0.15, self.base_accuracy - 0.18 * severity)
        is_correct = self._rng.random() < acc
        gold = options[0]  # by convention we store the correct option first
        if is_correct:
            ans = gold
        else:
            distractors = [o for o in options if o != gold] or [gold]
            ans = self._rng.choice(distractors)
        # confidence stays high regardless of severity (overconfidence)
        conf = min(1.0, max(0.0,
                            self._rng.gauss(self.overconfidence, 0.04)))
        return ans, conf


@dataclass
class HFVLMBackend:
    """
    Real Hugging Face VLM backend. Fill in when you have a GPU.

    Example models to try (small, open):
      - "Qwen/Qwen2-VL-2B-Instruct"
      - "Qwen/Qwen2-VL-7B-Instruct"
      - "OpenGVLab/InternVL2-2B"
      - "HuggingFaceTB/SmolVLM-Instruct"
    """
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cuda"
    _model: object = None
    _processor: object = None

    def load(self):
        # TODO (Week 3, on GPU):
        #   from transformers import AutoProcessor, AutoModelForImageTextToText
        #   self._processor = AutoProcessor.from_pretrained(self.model_id)
        #   self._model = AutoModelForImageTextToText.from_pretrained(
        #       self.model_id, torch_dtype="auto", device_map=self.device)
        raise NotImplementedError(
            "HFVLMBackend.load() is a stub. Fill in on a GPU machine in Week 3. "
            "Until then, use MockVLM to validate the pipeline."
        )

    def answer(self, image: Image.Image, question: str, options: list[str]):
        # TODO: build chat messages with the image + CONFIDENCE_PROMPT,
        #       run generation, decode text, then:
        #   return parse_answer_and_confidence(decoded_text)
        raise NotImplementedError("Fill in generation on GPU.")


def build_model(cfg: dict):
    """Factory: cfg['backend'] in {'mock','hf'}."""
    backend = cfg.get("backend", "mock")
    if backend == "mock":
        return MockVLM(
            seed=cfg.get("seed", 0),
            base_accuracy=cfg.get("base_accuracy", 0.82),
            overconfidence=cfg.get("overconfidence", 0.90),
        )
    elif backend == "hf":
        m = HFVLMBackend(model_id=cfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct"),
                         device=cfg.get("device", "cuda"))
        m.load()
        return m
    raise ValueError(f"Unknown backend '{backend}'")


if __name__ == "__main__":
    # Demonstrate parsing + the mock's overconfidence pattern.
    sample = "Reasoning: image looks fine.\nAnswer: cat\nConfidence: 88"
    print("Parse test:", parse_answer_and_confidence(sample))

    from PIL import Image as PImage
    img = PImage.new("RGB", (64, 64))
    model = MockVLM(seed=1)
    for sev in [0, 1, 2, 3]:
        img.info["severity"] = sev
        n, correct, confs = 300, 0, []
        for _ in range(n):
            ans, conf = model.answer(img, "what is this?", ["cat", "dog", "bird"])
            correct += (ans == "cat")
            confs.append(conf)
        print(f"severity={sev}: acc={correct/n:.2f}  mean_conf={sum(confs)/n:.2f}")
    print("\nSee: accuracy drops with severity, confidence stays ~0.90 (overconfident).")

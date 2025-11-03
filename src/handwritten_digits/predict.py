#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Handwritten Digits Recognition
File: predict.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-02
Updated: 2025-11-02
License: MIT License (see LICENSE file for details)
=

Description:
Single-image prediction CLI with auto pre-processing.

Usage:
hwr-predict --weights artifacts/model_latest.pt --image path/to/img.png

============================================================================
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image, ImageOps

from .model import MnistNet
from .utils import device as get_device, load_checkpoint


def preprocess_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")  # grayscale
    # Invert if white digit on black background
    if np.array(img).mean() < 127:
        img = ImageOps.invert(img)
    img = ImageOps.pad(img, (28, 28), method=Image.BILINEAR, color=0, centering=(0.5, 0.5))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    return tensor


@click.command()
@click.option("--weights", required=True, type=str)
@click.option("--image", required=True, type=str)
def main(weights: str, image: str) -> None:
    dev = get_device()
    model = MnistNet().to(dev)
    state = load_checkpoint(weights, map_location=dev)
    model.load_state_dict(state["model"])
    model.eval()

    x = preprocess_image(Path(image)).to(dev)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = int(prob.argmax(1).item())
        confidence = float(prob.max().item())
    click.echo(f"Prediction: {pred} (confidence={confidence:.2%})")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Handwritten Digits Recognition
File: test_data.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-02
Updated: 2025-11-02
License: MIT License (see LICENSE file for details)
=

Description:
Verify transforms and dataloader return expected shapes.

============================================================================
"""
from __future__ import annotations

from handwritten_digits.data import mnist_transforms


def test_transforms_output_shape():
    import torch
    from PIL import Image
    import numpy as np

    # Create a dummy 28x28 grayscale image
    arr = (np.random.rand(28, 28) * 255).astype("uint8")
    img = Image.fromarray(arr, mode="L")
    x = mnist_transforms()(img)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 28, 28)
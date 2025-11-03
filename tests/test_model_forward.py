#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Handwritten Digits Recognition
File: test_model_forward.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-02
Updated: 2025-11-02
License: MIT License (see LICENSE file for details)
=

Description:
Smoke test the forward pass shape.

============================================================================
"""
from __future__ import annotations

from handwritten_digits.model import MnistNet


def test_forward_shape(random_batch):
    model = MnistNet()
    out = model(random_batch)
    assert out.shape == (4, 10)
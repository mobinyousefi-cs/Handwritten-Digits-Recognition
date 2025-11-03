#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Handwritten Digits Recognition
File: conftest.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-02
Updated: 2025-11-02
License: MIT License (see LICENSE file for details)
=

Description:
PyTest fixtures.

============================================================================
"""
from __future__ import annotations

import torch
import pytest


@pytest.fixture(scope="session")
def random_batch():
    # batch of 4 fake MNIST-like images
    return torch.randn(4, 1, 28, 28)
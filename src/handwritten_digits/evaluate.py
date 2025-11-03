#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Handwritten Digits Recognition
File: evaluate.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-02
Updated: 2025-11-02
License: MIT License (see LICENSE file for details)
=

Description:
Model evaluation on the test set with accuracy and confusion matrix.

Usage:
hwr-eval --weights artifacts/model_latest.pt

============================================================================
"""
from __future__ import annotations

import click
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from rich.console import Console

from .config import DEFAULT_DATA_ROOT
from .data import mnist_loaders
from .model import MnistNet
from .utils import device as get_device, load_checkpoint

console = Console()


@click.command()
@click.option("--weights", required=True, type=str, help="Path to .pt checkpoint")
@click.option("--data-root", default=str(DEFAULT_DATA_ROOT), show_default=True)
@click.option("--batch-size", default=256, show_default=True, type=int)
@click.option("--num-workers", default=2, show_default=True, type=int)
def main(weights: str, data_root: str, batch_size: int, num_workers: int) -> None:
    dev = get_device()
    _, test_loader = mnist_loaders(data_root=data_root, batch_size=batch_size, num_workers=num_workers)

    model = MnistNet().to(dev)
    state = load_checkpoint(weights, map_location=dev)
    model.load_state_dict(state["model"])
    model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(test_loader.dataset)
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()

    console.rule("[bold cyan]Evaluation")
    console.print(f"Test loss: {avg_loss:.4f} | Accuracy: {acc:.2%}")
    console.print("\nClassification Report:\n" + classification_report(all_labels, all_preds, digits=4))
    console.print("Confusion Matrix:")
    console.print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
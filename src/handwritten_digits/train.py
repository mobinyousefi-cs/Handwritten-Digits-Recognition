#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Handwritten Digits Recognition
File: train.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-02
Updated: 2025-11-02
License: MIT License (see LICENSE file for details)
=

Description:
Training loop with validation logging and checkpointing.

Usage:
hwr-train --epochs 5 --batch-size 128 --lr 1e-3

============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from rich.console import Console

from .config import ARTIFACTS_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_DATA_ROOT, SEED
from .data import mnist_loaders
from .model import MnistNet
from .utils import device as get_device, save_checkpoint, seed_everything

console = Console()


@click.command()
@click.option("--epochs", default=DEFAULT_EPOCHS, show_default=True, type=int)
@click.option("--batch-size", default=DEFAULT_BATCH_SIZE, show_default=True, type=int)
@click.option("--lr", default=DEFAULT_LR, show_default=True, type=float)
@click.option("--data-root", default=str(DEFAULT_DATA_ROOT), show_default=True, type=str)
@click.option("--weights-out", default=str(ARTIFACTS_DIR / "model_latest.pt"), show_default=True, type=str)
@click.option("--num-workers", default=2, show_default=True, type=int)
@click.option("--seed", default=SEED, show_default=True, type=int)
def main(epochs: int, batch_size: int, lr: float, data_root: str, weights_out: str, num_workers: int, seed: int) -> None:
    seed_everything(seed)
    dev = get_device()

    train_loader, test_loader = mnist_loaders(data_root=data_root, batch_size=batch_size, num_workers=num_workers)

    model = MnistNet().to(dev)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    console.rule("[bold cyan]Training")
    best_acc = 0.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(dev), labels.to(dev)
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_train_loss = running_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)
        acc = correct / total
        console.print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} acc={acc:.4%}")

        # Checkpoint
        ckpt_path = Path(weights_out)
        save_checkpoint({"model": model.state_dict(), "acc": acc, "epoch": epoch}, ckpt_path)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint({"model": model.state_dict(), "acc": acc, "epoch": epoch}, ckpt_path.with_name("model_best.pt"))

    elapsed = time.time() - start
    console.rule(f"[bold green]Done in {elapsed:.1f}s | Best acc: {best_acc:.2%}")


if __name__ == "__main__":
    main()
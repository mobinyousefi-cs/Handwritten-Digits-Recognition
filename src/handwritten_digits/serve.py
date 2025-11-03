#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Handwritten Digits Recognition
File: serve.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-02
Updated: 2025-11-02
License: MIT License (see LICENSE file for details)
=

Description:
FastAPI service exposing /predict for MNIST digit inference.

Usage:
hwr-serve --weights artifacts/model_latest.pt --host 0.0.0.0 --port 8000

============================================================================
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from .model import MnistNet
from .predict import preprocess_image
from .utils import device as get_device, load_checkpoint


app = FastAPI(title="Handwritten Digits Recognition API", version="0.1.0")
model: Optional[MnistNet] = None
_dev = get_device()


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("L")  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    # Save to temp Path-like in memory
    from io import BytesIO

    buf = BytesIO(contents)
    buf.seek(0)
    x = preprocess_image(Path("/dev/stdin")).to(_dev)  # dummy path for interface
    # but preprocess_image expects a Path; when serving, we replicate its steps directly:
    # Re-implement quick path to avoid disk writes
    from PIL import ImageOps
    import numpy as np

    img = Image.open(buf).convert("L")
    if np.array(img).mean() < 127:
        img = ImageOps.invert(img)
    img = ImageOps.pad(img, (28, 28), method=Image.BILINEAR, color=0, centering=(0.5, 0.5))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.1307) / 0.3081
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(_dev)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = int(prob.argmax(1).item())
        confidence = float(prob.max().item())
    return JSONResponse({"digit": pred, "confidence": confidence})


@click.command()
@click.option("--weights", required=True, type=str)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--reload/--no-reload", default=False, show_default=True)
@click.option("--workers", default=1, show_default=True, type=int)
def main(weights: str, host: str, port: int, reload: bool, workers: int) -> None:
    global model
    model = MnistNet().to(_dev)
    state = load_checkpoint(weights, map_location=_dev)
    model.load_state_dict(state["model"])
    model.eval()
    uvicorn.run(app, host=host, port=port, reload=reload, workers=workers)


if __name__ == "__main__":
    main()
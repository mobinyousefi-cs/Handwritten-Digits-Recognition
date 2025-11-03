# Handwritten Digits Recognition (MNIST)

A clean, production‑ready PyTorch implementation using the `src/` layout, CI, and console entry points. Trains a small CNN on MNIST, evaluates with a confusion matrix, serves predictions via FastAPI, and exposes CLIs:

- `hwr-train` – train on MNIST and save a model to `artifacts/`
- `hwr-eval` – evaluate on the test set and print metrics
- `hwr-predict` – predict a single image from disk
- `hwr-serve` – run a FastAPI server for HTTP predictions

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt

# Train
hwr-train --epochs 5 --batch-size 128 --lr 1e-3

# Evaluate
hwr-eval --weights artifacts/model_latest.pt

# Predict a single image (28x28 grayscale or auto-preprocessed)
hwr-predict --weights artifacts/model_latest.pt --image path/to/image.png

# Serve API
hwr-serve --weights artifacts/model_latest.pt --host 0.0.0.0 --port 8000
```

## Dataset
Default: `torchvision.datasets.MNIST` (auto‑download from Yann LeCun’s site/mirrors). Original page: http://yann.lecun.com/exdb/mnist/

## Repo Standards
- `src/` layout, `pyproject.toml` (PEP 621)
- Type hints, docstrings, deterministic seeding
- Linters: Ruff + Black; Tests: PyTest; CI: GitHub Actions
- MIT License

## Docker
```bash
docker build -t hwr:latest .
docker run --rm -p 8000:8000 hwr:latest hwr-serve --weights /app/artifacts/model_latest.pt
```
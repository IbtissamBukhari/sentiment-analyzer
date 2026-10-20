"""Download the quantized ONNX model and tokenizer into models/.

Run once before starting the API locally:  python -m api.download_model
The Docker build runs this automatically.
"""

import sys
import urllib.request
from pathlib import Path

from api.classifier import MODEL_DIR

REPO = "https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest/resolve/main"
FILES = {
    "onnx/model_quantized.onnx": "model_quantized.onnx",
    "tokenizer.json": "tokenizer.json",
    "config.json": "config.json",
}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        target = MODEL_DIR / local
        if target.exists():
            print(f"{local}: already present")
            continue
        print(f"Downloading {remote} ...", flush=True)
        urllib.request.urlretrieve(f"{REPO}/{remote}", target)
    print(f"Model files are in {MODEL_DIR}")


if __name__ == "__main__":
    sys.exit(main())

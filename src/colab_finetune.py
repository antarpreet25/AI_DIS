"""
colab_finetune.py

The ZEDD v3 fine-tuning step, extracted so it can run on a Colab GPU
(minutes) instead of a local CPU (hours). This is the SAME contrastive
objective and label convention as ZEDDTrainer.fine_tune() — only the
device and batch size differ.

It is safe to run locally too (it will just use CPU and be slow):

    python src/colab_finetune.py data/v3/train_pairs.json data/zedd_finetuned_model_v3

On Colab you normally paste the cells from COLAB_GUIDE.md rather than
uploading this file, but the logic here is the canonical reference.

Input  : train_pairs.json  — list of {"text1","text2","label"} where
         label 0 = dissimilar (injected vs clean), label 1 = similar
         (clean vs clean). Produced by `zedd_trainer.py --prepare-v3`.
Output : a SentenceTransformer model directory.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


EPOCHS = 6          # plenty on GPU; loss plateaus by ~epoch 3
BATCH_SIZE = 32     # GPU-sized; drop to 8 if you run this on CPU
LEARNING_RATE = 2e-5
BASE_MODEL = "all-MiniLM-L6-v2"


def load_examples(pairs_path: str):
    from sentence_transformers import InputExample

    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    examples = []
    for p in pairs:
        t1 = p.get("text1") or p.get("injected")
        t2 = p.get("text2") or p.get("clean")
        if not t1 or not t2:
            continue
        examples.append(InputExample(texts=[t1, t2], label=float(p["label"])))

    import random
    random.Random(42).shuffle(examples)
    n_pos = sum(1 for e in examples if e.label == 1.0)
    print(f"[colab_finetune] {len(examples)} examples "
          f"({len(examples) - n_pos} dissimilar / {n_pos} similar)")
    return examples


def finetune(pairs_path: str, out_dir: str,
             epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> None:
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    examples = load_examples(pairs_path)

    model = SentenceTransformer(BASE_MODEL)
    try:
        dev = model.device
    except Exception:
        dev = "unknown"
    print(f"[colab_finetune] base model on device: {dev}")

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss_fn = losses.ContrastiveLoss(model)
    warmup = int(len(loader) * epochs * 0.1)

    t0 = time.time()
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": LEARNING_RATE},
        show_progress_bar=True,
    )
    print(f"[colab_finetune] trained {epochs} epochs in {time.time() - t0:.0f}s")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save(out_dir)
    print(f"[colab_finetune] saved -> {out_dir}")


if __name__ == "__main__":
    pairs = sys.argv[1] if len(sys.argv) > 1 else "data/v3/train_pairs.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/zedd_finetuned_model_v3"
    finetune(pairs, out)

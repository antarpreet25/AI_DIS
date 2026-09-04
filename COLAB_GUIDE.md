# Fine-tuning ZEDD v3 on Google Colab — step by step

You do **not** need to know anything about Colab. Follow this top to bottom.

**What this achieves:** the ZEDD model needs to be fine-tuned on ~3,600
training pairs. On your PC's CPU that takes hours. On Colab's free GPU it
takes a few minutes. You generate the data on your PC, train on Colab,
bring the model back, and finish on your PC.

**What stays on your PC (fast, already works):** generating the data,
building the baseline, calibrating thresholds, validating, promoting.
**Only the fine-tune goes to Colab.**

Nothing here overwrites your current working model until the very last
step (`--promote-v3`), and that step backs up what you have first.

---

## Part 1 — Generate the data (on your PC, in VS Code)

Open the VS Code terminal (menu **Terminal → New Terminal**) and run:

```
python src/zedd_trainer.py --prepare-v3
```

This creates a `data/v3/` folder containing:

- `train_pairs.json`  ← the only file you upload to Colab
- `calibration_set.json`  ← used later, on your PC, for leakage-free threshold calibration
- `baseline_corpus_meta.json`  ← just records the settings

Confirm the file exists: in VS Code's left-hand **Explorer** panel, expand
`data` → `v3`, you should see `train_pairs.json` (a few MB).

If you already ran `--prepare-v3`, you can skip this part.

---

## Part 2 — Open Colab and turn on the GPU

1. In a web browser go to **https://colab.research.google.com**
2. Sign in with your Google account if asked.
3. A dialog titled "Open notebook" appears. Click **New notebook** (bottom
   left of that dialog). If no dialog appears: menu **File → New notebook**.
4. You now see an empty page with one grey box (a "cell") and a `▶` play
   button on its left.
5. Turn on the GPU: menu **Runtime → Change runtime type**.
   - Under **Hardware accelerator** choose **T4 GPU**.
   - Click **Save**.

That's the setup. Now you'll add four cells and run them in order. To add
a new cell, hover just below the current cell and click **+ Code**, or
press the **+ Code** button at the top left.

---

## Part 3 — Cell 1: check the GPU and install the library

Click the first (empty) cell, paste this in, then click its `▶` button
(or press **Shift+Enter**):

```python
!nvidia-smi -L          # should print something like "GPU 0: Tesla T4"
!pip -q install "sentence-transformers>=2.7,<4"
print("ready")
```

Wait until it prints `ready` (the install takes ~1 minute the first time).
If `nvidia-smi` says "command not found" or shows no GPU, go back to
Part 2 step 5 — the GPU is not enabled.

---

## Part 4 — Cell 2: upload your training pairs

Add a new cell, paste this, run it:

```python
from google.colab import files
uploaded = files.upload()      # a "Choose Files" button appears — click it
print("uploaded:", list(uploaded.keys()))
```

When the **Choose Files** button appears, click it, and in the file
picker navigate to your project folder →  `data` → `v3` → select
**`train_pairs.json`** → Open.

Wait for it to finish uploading (you'll see a progress line, then
`uploaded: ['train_pairs.json']`).

---

## Part 5 — Cell 3: fine-tune (this is the part that needs the GPU)

Add a new cell, paste this **exactly**, run it:

```python
import json, random, time
from pathlib import Path
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

PAIRS   = "train_pairs.json"
OUT_DIR = "zedd_finetuned_model_v3"
BASE    = "all-MiniLM-L6-v2"
EPOCHS, BATCH, LR = 6, 32, 2e-5

pairs = json.loads(Path(PAIRS).read_text(encoding="utf-8"))
examples = [
    InputExample(texts=[p["text1"], p["text2"]], label=float(p["label"]))
    for p in pairs if p.get("text1") and p.get("text2")
]
random.Random(42).shuffle(examples)
n_pos = sum(1 for e in examples if e.label == 1.0)
print(f"{len(examples)} examples ({len(examples)-n_pos} dissimilar / {n_pos} similar)")

model = SentenceTransformer(BASE)
print("device:", model.device)          # should say cuda:0
loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
loss_fn = losses.ContrastiveLoss(model)

t0 = time.time()
model.fit(
    train_objectives=[(loader, loss_fn)],
    epochs=EPOCHS,
    warmup_steps=int(len(loader) * EPOCHS * 0.1),
    optimizer_params={"lr": LR},
    show_progress_bar=True,
)
print(f"trained in {time.time()-t0:.0f}s")
model.save(OUT_DIR)
print("saved ->", OUT_DIR)
```

A progress bar runs. On a T4 GPU this finishes in roughly **3–6 minutes**.
`device: cuda:0` confirms it's using the GPU — if it says `cpu`, stop,
fix the GPU (Part 2 step 5), and re-run this cell.

---

## Part 6 — Cell 4: zip and download the model

Add a new cell, paste, run:

```python
import shutil
shutil.make_archive("zedd_finetuned_model_v3", "zip", "zedd_finetuned_model_v3")
from google.colab import files
files.download("zedd_finetuned_model_v3.zip")
```

Your browser downloads **`zedd_finetuned_model_v3.zip`** (~90 MB) to your
usual Downloads folder.

You're done with Colab. You can close the tab. (Optional: menu
**Runtime → Disconnect and delete runtime** to free it.)

---

## Part 7 — Put the model back into the project (on your PC)

1. Find `zedd_finetuned_model_v3.zip` in your **Downloads** folder.
2. Extract it. On Windows: right-click → **Extract All…** → extract to a
   temporary place.
3. You need the folder **`zedd_finetuned_model_v3`** — the one that
   *directly contains* `config.json`, `model.safetensors`, `modules.json`,
   `tokenizer.json`, a `1_Pooling` folder, etc.
   - Sometimes extraction nests it (e.g.
     `zedd_finetuned_model_v3/zedd_finetuned_model_v3/config.json`). If so,
     use the **inner** folder — the one with `config.json` right inside it.
4. Move/copy that folder into your project's `data/` folder so the final
   path is exactly:

   ```
   AI_DIS/data/zedd_finetuned_model_v3/
       config.json
       model.safetensors
       modules.json
       tokenizer.json
       1_Pooling/
       ...
   ```

   In VS Code's Explorer you can just drag the extracted
   `zedd_finetuned_model_v3` folder onto the `data` folder.

---

## Part 8 — Finish locally (fast, no GPU)

In the VS Code terminal:

```
python src/zedd_trainer.py --finish-v3
```

This will, using your PC:

1. load the v3 model you just placed,
2. build the v3 **baseline** from the balanced templated document corpus,
3. **calibrate** the three per-mode thresholds **on the disjoint
   calibration set** (`data/v3/calibration_set.json`) — never on the
   evaluation set, which is the data-leakage fix,
4. print a **validation table** comparing three configs — your old v1
   backup, the current production model, and v3 — on both the evaluation
   set and a disjoint holdout.

It writes `data/zedd_finetuned_model_v3/`, `..._baseline_v3.json`,
`..._calibration_v3.json`. **It does not touch production yet.**

Read the `holdout` columns — that's the honest comparison (no config was
trained or calibrated on that data).

---

## Part 9 — Promote v3, or roll back

If v3 looks good (holdout recall up on sentence/dual_encoder, document
mode not meaningfully down, FPR ~0):

```
python src/zedd_trainer.py --promote-v3
```

This copies v3 over the production files and backs up the current ones as
`*_pre_v3_backup`. From then on the pipeline (`python src/evaluator.py
--mode=...`) uses v3.

If you ever want to undo it, the backups are right there in `data/`:
- `data/zedd_finetuned_model_pre_v3_backup/`
- `data/zedd_finetuned_baseline_pre_v3_backup.json`
- `data/zedd_calibration_pre_v3_backup.json`

Copy those back over the production names to revert.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cell 3 prints `device: cpu` | GPU not enabled — Part 2 step 5, then re-run Cell 3. |
| `nvidia-smi: command not found` | Same — GPU not enabled. |
| Colab says "you are not connected to a runtime" | Menu **Runtime → Connect**, wait for the green tick, re-run cells from Cell 1. |
| Upload button never appears in Cell 2 | Re-run the cell; if still nothing, refresh the page and start again from Cell 1. |
| Colab disconnects mid-training | Free Colab has time limits. Re-run from Cell 1. If it keeps happening, lower `EPOCHS` to 4 in Cell 3. |
| `--finish-v3` says "v3 model not found" | The folder isn't at `data/zedd_finetuned_model_v3/` with `config.json` directly inside it — see Part 7 step 3. |
| Want a quicker test first | In Cell 3 set `EPOCHS = 2`. Fine for a smoke test; use 6 for the real run. |

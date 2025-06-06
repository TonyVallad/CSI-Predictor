Below is a ready-to-use **five-stage prompt set**.
Feed each one to Claude-4 Sonnet in Cursor, **one after the other**, letting it generate / modify the code before moving to the next stage.
(The stages are independent, so you can stop early or iterate on any single step.)

---

## Stage 1 – Scaffold & Configuration

> **Prompt 1 – PROJECT SKELETON**
> You are creating a modular PyTorch project that predicts 6-zone CSI scores on chest X-rays.
> **Create the folder / file scaffold and all *non-Python* config assets** exactly as described below.
>
> 1. **Folder tree**
>
>    ```
>    CSI-PREDICTOR/ (root folder of this project)
>    ├── docs/
>    ├── notebooks/
>    ├── src/
>    │   ├── __init__.py
>    │   ├── config.py
>    │   ├── data.py
>    │   ├── models/
>    │   │   ├── __init__.py
>    │   │   ├── backbones.py
>    │   │   └── head.py
>    │   ├── train.py
>    │   ├── evaluate.py
>    │   └── utils.py
>    ├── .env
>    ├── config.ini
>    ├── main.py (file where we can run the whole process following parameters set in these files: .env, config.ini, config.py)
>    ├── requirements.txt
>    ├── README.md
>    └── .gitignore
>    ```
> 2. **`.env` template** containing the keys
>    `DEVICE, LOAD_DATA_TO_MEMORY, DATA_SOURCE, DATA_PATH, MODELS_PATH, CSV_PATH, INI_PATH, LABELS_CSV, LABELS_CSV_SEPARATOR`.
> 3. **`config.ini` template** with sensible default values for
>    `BATCH_SIZE, N_EPOCHS, PATIENCE, LEARNING_RATE, OPTIMIZER, MODEL_ARCH, MODEL_PATH`.
> 4. Add **`.gitignore`** for Python, virtual-env, PyCharm, VSCode, Jupyter, and WandB artifacts.
> 5. Create **`requirements.txt`** listing: `torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `python-dotenv`, `loguru`, `tqdm`, `wandb`, `Pillow`.
> 6. Produce a concise **`README.md`** explaining setup, config files, training command, and eval command.

Make sure the file contents are emitted verbatim in Markdown code-blocks that Cursor can place into the respective files.

---

## Stage 2 – Centralised Configuration Loader

> **Prompt 2 – CONFIG MODULE**
> Write **`src/config.py`** that:
>
> * loads `.env` via `dotenv_values`;
> * loads `config.ini` via `configparser`;
> * validates + merges them into a single immutable dataclass `Config`;
> * exposes a singleton `cfg = Config()` so other modules can simply `from src.config import cfg`;
> * handles type conversions (`int`, `float`, `bool`) automatically;
> * copies the resolved `config.ini` into `INI_PATH` with a timestamp when training starts;
> * logs any missing keys using `loguru`.

Include full docstrings and unit-test stubs in inline comments.

---

## Stage 3 – Data Pipeline

> **Prompt 3 – DATA PIPELINE**
> Build **`src/data.py`** and update `src/utils.py` with helper functions.
> Requirements:
>
> * **CSV ingestion** using the path/filename from `cfg`, loading columns:
>   `FileID, right_sup, left_sup, right_mid, left_mid, right_inf, left_inf`.
> * Convert NaNs in label columns into an **“unknown / ungradable” class index 4**.
> * **Split** into train/val/test using `sklearn.model_selection.train_test_split` with stratification over *presence* of unknown scores to keep class balance.
> * **Torch-compatible Dataset / DataLoader**:
>
>   * Accepts `transform` argument (use `torchvision.transforms`); default resize to model’s expected size (read from a dict keyed by `cfg.MODEL_ARCH`), convert grayscale to 3-channel, then `ToTensor` and per-channel mean/std normalisation.
>   * Loads images lazily from disk **unless** `cfg.LOAD_DATA_TO_MEMORY==True`, in which case it pre-caches tensors.
>   * Returns: `image_tensor, label_tensor` where `label_tensor` is shape `(6,)`, each value in `{0,1,2,3,4}`.
> * Include a `show_batch()` utility that randomly samples a batch and shows the 6 predicted zone masks side-by-side in a single matplotlib figure (for notebooks/debug-only).
> * Wrap all loops with `tqdm`.

---

## Stage 4 – Model, Loss & Training Loop

> **Prompt 4 – MODEL + TRAINING**
> Implement / update the following files:
>
> 1. **`src/models/backbones.py`**
>
>    * Provide factory `get_backbone(name, pretrained=True)` supporting:
>
>      * `"ResNet50"` (torchvision),
>      * `"CheXNet"` (DenseNet121 with ImageNet weights and first conv changed for 3-chan XR),
>      * `"Rad_DINO"` (placeholder commented out),
>      * `"Custom_1"` (simple 5-layer CNN as baseline).
> 2. **`src/models/head.py`**
>
>    * Define `CSIHead(backbone_out_dim, n_classes_per_zone=5)` that:
>
>      * sets six parallel `nn.Linear(backbone_out_dim, n_classes_per_zone)` heads;
>      * in `forward`, returns a tensor `(B, 6, n_classes_per_zone)`.
> 3. **`src/models/__init__.py`**
>
>    * Expose `build_model(cfg)` that stitches backbone+head and moves to `cfg.DEVICE`.
> 4. **`src/train.py`**
>
>    * Parse `cfg`, seed RNGs, instantiate data loaders and model.
>    * Compute **masked Cross-Entropy loss** that ignores zone positions where ground-truth == 4.
>    * Log to **Weights & Biases**:
>
>      * Config parameters, LR schedule, and every epoch’s train/val loss, macro-averaged F1, and per-zone F1.
>    * Use `loguru` for file + console logging.
>    * Early stopping on `cfg.PATIENCE`.
>    * Save best model to `cfg.MODEL_PATH/model_name.pth`, where
>      `model_name = f"{cfg.MODEL_ARCH} - {datetime.now():%Y-%m-%d %H:%M:%S}"`.
> 5. Add CLI entry-points:
>
>    ```
>    python -m src.train
>    python -m src.evaluate
>    ```
>
>    Accept `--ini path/to/other_config.ini` override.

Provide complete code plus inline explanations **only where necessary**.

---

## Stage 5 – Evaluation & Utility Scripts

> **Prompt 5 – EVALUATION, LOGGING & MISC**
>
> 1. **`src/evaluate.py`**
>
>    * Loads the saved model, runs on the test set, logs per-zone confusion matrices and classification reports to WandB, and prints aggregated metrics.
> 2. Extend **`src/utils.py`** with:
>
>    * `make_run_name(cfg)` helper (same timestamped pattern as in training).
>    * `seed_everything(seed)` using `torch`, `numpy`, `random`.
>    * pretty-printer for `cfg`.
> 3. Add **Loguru** sink setup (rotating file handler) in `utils.py` and import it in all main modules.
> 4. Update the **README** with usage examples for training, resuming, and evaluation; plus links to WandB dashboards.

---

### How to use the prompt set

1. Copy **Prompt 1** into Cursor, run Claude, let it write all scaffold files.
2. Inspect / commit.
3. Proceed with **Prompt 2**, etc.
4. Iterate or rerun any stage if you want refinements (e.g. to add a new backbone you’d edit Stage 4’s prompt).

Each prompt is self-contained and instructs Claude to **emit full file contents** inside Markdown code-blocks, making it straightforward for Cursor to save them.

Feel free to tweak hyper-parameters or folder names in the prompts before sending.
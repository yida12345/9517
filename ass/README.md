# Comparing different deep learning models: Segmentation of standing dead trees in aerial images of forests

> COMP9517: Group Project

---

## Group member
  - Tingrui Zhang (z5616445)
  - Mochi Zhang (z5493308)
  - Shuochen Zhu (z5507779)
  - Zhongwei Kou (z546772)
  - Shuchang Xing (z5500862)

---

## Highlights

- **Multi‑spectral input (5 channels):** RGB **+** NRG are merged at load time and fed to networks (U‑Net initialized with `n_channels=5`; DeepLab’s first conv replaced to accept 5 channels).
- **TensorBoard‑first workflow:** Training scripts are adapted to log scalars and samples for both models.

---

## Project Structure

```text
.
├── Dataset/
│   ├── cut.py              # Preprocess/slice raw images (crop/split)
│   ├── show.py             # Visualize images & masks (quick inspection)
│   ├── test.py             # Load a test split and evaluate models
│   └── trans.py            # Resizing / padding transforms to standardize inputs
├── UNet/
│   ├── unet/               # UNet architecture: unet_model.py, unet_parts.py
│   ├── utils/              # data_loading.py (BasicDataset), dice_score.py, utils.py
│   ├── train_tensorboard.py
│   ├── evaluate.py
│   ├── predict_test.py
│   └── runs/               # TensorBoard event logs for UNet
└── DeepLabV3Plus/
    ├── utils/              # ext_transforms.py, loss.py, scheduler.py, utils.py
    ├── metrics/            # stream_metrics.py (IoU, pix acc)
    ├── main_tensorboard.py # Training entry point for DeepLabV3+
    ├── test.py             # Validation/test visualization & metrics
    ├── predict.py          # Inference on image or folder
    ├── runs/               # TensorBoard event logs for DeepLab
    └── checkpoints/        # Saved weights (best/latest)

```

### What each folder/file is for

- **Dataset/**
  - `cut.py` – quick slicing/cropping for raw images.
  - `show.py` – visualize samples + masks for sanity checks.
  - `test.py` – test‑time loading/evaluation helpers.
  - `trans.py` – input standardization (resize/pad while preserving aspect ratio).

- **UNet/**
  - `unet/` - From https://github.com/milesial/Pytorch-UNet
    - `unet_model.py` – full U‑Net assembly.
    - `unet_parts.py` – building blocks (DoubleConv, Down, Up, etc.).
  - `utils/`
    - `data_loading.py` – **BasicDataset** merges **RGB** and **NRG** into a 5‑channel tensor; returns image/mask pairs.
    - `dice_score.py` – Dice coefficient utilities.
    - `utils.py` – misc. helpers (logging, checkpoints). From https://github.com/milesial/Pytorch-UNet
  - `train_tensorboard.py` – train U‑Net with CLI args and TensorBoard logging.
  - `evaluate.py` – compute validation metrics (e.g., Dice).
  - `predict_test.py` – batch inference over a test folder; saves predicted masks.

- **DeepLabV3Plus/**
  - `main_tensorboard.py` – train DeepLabV3+ (ResNet/MobileNet/HRNet/Xception backbones) with TensorBoard logging.
  - `test.py` – evaluation & visualization (Matplotlib).
  - `predict.py` – inference on one or many images. From https://github.com/VainF/DeepLabV3Plus-Pytorch
  - `utils/` – data transforms, losses (e.g., CE/Focal), schedulers, misc. utilities.
  - `runs/` – TensorBoard logs
  - `checkpoints/` – saved weights.

- **Root**
  - `README.md` – this file.



---

## Dataset Preparation

```text
.
├── NRG_images
│   ├── train
│   │   ├── ar037_2019_n_06_04_0.png
│   │   └── ar037_2019_n_07_05_0.png
│   └── val
│       ├── ar037_2019_n_08_14_0.png
│       └── ar039_2019_n_05_10_0.png
├── RGB_images
│   ├── train
│   │   ├── ar037_2019_n_06_04_0.png
│   │   └── ar037_2019_n_07_05_0.png
│   └── val
│       ├── ar037_2019_n_08_14_0.png
│       └── ar039_2019_n_05_10_0.png
└── masks
    ├── train
    │   ├── ar037_2019_n_06_04_0.png
    │   └── ar037_2019_n_07_05_0.png
    └── val
        ├── ar037_2019_n_08_14_0.png
        └── ar039_2019_n_05_10_0.png
```

**Important:**
- Filenames must align across `RGB_images`, `NRG_images`, and `masks` so each ID matches up.
- If you don’t have a split, either create `train/` & `val/` folders.

---

## How to Run

### 1) U‑Net (training, evaluation, prediction)

```bash
cd UNet

# Train (50 epochs, bs=8, LR=1e-5; AMP enabled)
python train_tensorboard.py --epochs 50 --batch-size 8 --learning-rate 1e-5 --amp

# Evaluate a trained checkpoint
python evaluate.py --model path/to/checkpoint.pth

# Predict on a test set and save masks
python predict_test.py --checkpoint path/to/best_model.pth --out-dir path/to/results/
```

TensorBoard logs are written under `UNet/runs/`. Launch TensorBoard to monitor curves and sample outputs:

```bash
tensorboard --logdir UNet/runs
```

### 2) DeepLabV3+ (training, evaluation, inference)

```bash
cd DeepLabV3Plus

# Train (ResNet‑50 backbone, 30k iters, output_stride=16, GPU 0)
python main_tensorboard.py   --model deeplabv3plus_resnet50   --batch_size 16   --lr 0.01   --output_stride 16   --gpu_id 0   --total_itrs 30000

# Evaluate
python test.py --checkpoint checkpoints/best_deeplabv3plus_resnet50.pth

# Inference on an image folder
python predict.py   --input ./some_images/   --model deeplabv3plus_resnet50   --ckpt checkpoints/best_model.pth   --dataset custom   --save_val_results_to ./predictions
```

Checkpoints are saved under `DeepLabV3Plus/checkpoints/`. Training curves live in `DeepLabV3Plus/runs/`.

---

## How this differs from the official repositories

- **Multi‑spectral input end‑to‑end:** This project **extends the original code** by (1) merging RGB+NRG in the dataset and (2) setting the networks to **5‑channel** input (U‑Net `n_channels=5`; DeepLab’s `conv1` replaced).  
- **Custom dataset instead of VOC/Cityscapes:** Training is tailored to your folder structure (RGB/NRG/masks) rather than the stock VOC/Cityscapes pipelines.
- **TensorBoard over W&B/Visdom:** Monitoring is unified via TensorBoard; related flags from the originals may still exist but TensorBoard is the primary path.
- **Script organization for clarity:** `train.py → train_tensorboard.py`, `main.py → main_tensorboard.py`; DeepLab adds a dedicated `test.py` for evaluation/visualization.
- **Defaults chosen for teaching:** Example defaults include a **20% validation split** and **AMP enabled** to speed up training on modern GPUs.

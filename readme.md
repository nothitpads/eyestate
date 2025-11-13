# Astana IT University — Computer Vision

## Assignment 2 (Continuation): “Level-Up Your A1 Topic with SOTA CV + Demo”

**Student:** Ansar Shangilov (SE-2311)
**Instructor:** Baimukanova Zhanerke

**Deliverables:**

* Web page or mobile app
* Report (6–8 pages)
* 3–5 minute demo video
* GitHub repository

---

# Real-Time Eye State Recognition

![ezgif-68e2c2a89fa08409](https://github.com/user-attachments/assets/6200e60a-3cbb-4813-8c0b-4c826724fffc)


## Abstract

I trained an EfficientNetV2-S model to classify eye crops as **open** or **closed**.
Dataset size: **~80k MRL images**, balanced.
Final test accuracy: **~96%**.

Augmentations included standard transformations plus a custom occlusion patch.
Grad-CAM was used for explainability.
The final model was exported to **ONNX** for browser deployment.

---

# Dataset

Two labels: **open**, **closed**.
Data sourced from the **MRL dataset** with additional augmentations.
Training/validation/testing splits were created after balancing.

* Total samples: ~80k

### Augmentations (training only)

* Horizontal flip
* Rotation (±15°)
* Brightness/contrast jitter
* Slight Gaussian blur
* Rectangular occlusion over eyelid (simulated glasses glare / obstruction)

These operations improved robustness to lighting, motion, and noise.

---

# SOTA Model: EfficientNetV2-S

EfficientNetV2-S is a modern CNN architecture combining:

* **MBConv** blocks (lightweight, depthwise convolutions)
* **Fused-MBConv** blocks (faster on modern hardware)

This mix produces strong accuracy–speed tradeoffs.

Final classifier:

```
Linear(out_features=2)
```

The model works well for eye-state classification because eye crops are small and require subtle feature extraction under blur or low-resolution conditions.

---

# Implementation

* Start from ImageNet-pretrained EfficientNetV2-S
* Replace classifier with 2-unit linear layer
* Train in two phases:

  * Phase 1: freeze early layers
  * Phase 2: unfreeze + fine-tune with lower LR
* Optimizer: **AdamW**
* LR scheduler: **cosine annealing**
* Early stopping on **macro-F1**
* Export best checkpoint to ONNX
* Apply **dynamic quantization** for faster inference in CPU/WEB environments

---

# Experiments

### Baseline: simple CNN

### Advanced: EfficientNetV2-S

| Model            | Acc  | Prec | Rec  | Macro-F1 | CPU Inference        | Size                          |
| ---------------- | ---- | ---- | ---- | -------- | -------------------- | ----------------------------- |
| Baseline CNN     | 0.89 | 0.88 | 0.89 | 0.88     | 4–6 ms               | ~1.2 MB                       |
| EfficientNetV2-S | 0.96 | 0.96 | 0.95 | 0.96     | 12–18 ms (quantized) | ~14–16 MB full, ~5–6 MB quant |

---

# Ethics and Limitations

* Models trained on limited eye shapes or lighting conditions may generalize poorly to other groups.
* Glasses glare, makeup, extreme angles, and low light reduce reliability.
* Not suitable for surveillance or fatigue monitoring without explicit consent.
* Underrepresentation of certain demographic groups in datasets causes bias.
* Should not be used for high-stakes decisions; label system clearly as automated.

Disclosure: augmentations are synthetic variations only (no GAN). If synthetic images are added, they must be watermarked.

---

# Testing Manual

### Environment Example

* Windows
* Python 3.11
* PyTorch 2.x
* timm
* onnx / onnxruntime / onnxruntime-tools
* torchvision
* numpy
* opencv-python
* pillow

---

# Training

```
python training/train.py \
  --model efficientnetv2_s \
  --pretrained \
  --num-classes 2 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --augmentations "flip,rotate,brightness,contrast,blur,occlusion" \
  --out-dir weights/
```

# Evaluation

```
python eval.py \
  --weights best_model.pth \
  --data data/test \
  --batch-size 64
```

# Quantization

```
python -m onnxruntime.tools.convert_onnx_models_to_onnx \
  --input eye_state.onnx \
  --output eye_state_quant.onnx \
  --quantization dynamic
```

# Running

```
python server-predict.py
python -m http.server 3000
```



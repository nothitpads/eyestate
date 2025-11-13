# test_model_local.py
from PIL import Image
import torch
import torchvision.transforms as T
from train import build_model
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MODELS_DIR = (ROOT / "../models").resolve()
CKPT = MODELS_DIR / "best_model.pth"
IMG = MODELS_DIR / "last_recv.jpg"

INPUT_SIZE = 80
NUM_CLASSES = 2
CBAM = True
CLASS_LABELS = ["CLOSED", "OPEN"]

# Build model
try:
    model = build_model(num_classes=NUM_CLASSES, cbam=CBAM, pretrained=False, input_dummy_size=INPUT_SIZE)
except TypeError:
    model = build_model(num_classes=NUM_CLASSES, cbam=CBAM, pretrained=False)
model.eval()

# Robust checkpoint loader: find the real state_dict inside common checkpoint containers
if not CKPT.exists():
    print("Checkpoint not found:", CKPT)
    sys.exit(1)

ckpt = torch.load(str(CKPT), map_location="cpu")

# Determine where the actual weights live
state = None
if isinstance(ckpt, dict):
    # common candidate keys
    for candidate in ("model_state", "state_dict", "model_state_dict", "model_state_dicts", "model", "model_state_dict_epoch"):
        if candidate in ckpt and isinstance(ckpt[candidate], dict):
            state = ckpt[candidate]
            print(f"Using checkpoint key for weights: '{candidate}'")
            break

    if state is None:
        # try nested dict heuristic: find first dict whose keys look like parameter names
        found = None
        for k, v in ckpt.items():
            if isinstance(v, dict):
                sample_keys = list(v.keys())[:5]
                if sample_keys and all(isinstance(sk, str) and '.' in sk for sk in sample_keys):
                    found = v
                    print(f"Using nested dict under checkpoint key: '{k}' as weights")
                    break
        if found is not None:
            state = found

    if state is None:
        # fallback: if ckpt itself looks like a state_dict (keys contain dots), use it
        sample_keys = list(ckpt.keys())[:5]
        if sample_keys and all(isinstance(sk, str) and '.' in sk for sk in sample_keys):
            state = ckpt
            print("Checkpoint appears to be a raw state_dict and will be used directly.")
        else:
            # last resort: print top-level keys and fail
            print("Unable to locate model state_dict inside checkpoint. Top-level keys:", list(ckpt.keys())[:50])
            raise RuntimeError("No model state_dict found in checkpoint.")
else:
    # ckpt is already a state dict (rare)
    state = ckpt

# Strip module. prefix if present and load
new_state = {}

# Insert before model.load_state_dict(...)
print("---- Checkpoint param shapes (first 40) ----")
for k, v in list(new_state.items())[:40]:
    print(k, tuple(v.shape))
print("\n---- Model param shapes (first 40) ----")
ms = model.state_dict()
for k in list(ms.keys())[:40]:
    print(k, tuple(ms[k].shape))


for k, v in state.items():
    new_state[k.replace("module.", "")] = v

try:
    model.load_state_dict(new_state, strict=False)
    print("Loaded weights into model with strict=False")
except Exception as e:
    print("Warning: model.load_state_dict failed. Exception:", e)
    raise

# Transform (RGB)
transform = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

if not IMG.exists():
    print("No test image found at", IMG)
    sys.exit(1)

img = Image.open(str(IMG)).convert("RGB")
print("Loaded test image:", IMG, "size:", img.size)
# show image if environment supports it
try:
    img.show()
except Exception:
    pass

# Prepare tensors
t_rgb = transform(img).unsqueeze(0)  # 1xCxHxW (RGB)
# BGR by swapping channels
t_bgr = t_rgb.clone()[:, [2,1,0], :, :]  # swap RGB->BGR

with torch.no_grad():
    out_rgb = model(t_rgb)
    probs_rgb = torch.softmax(out_rgb, 1).numpy()[0].tolist()
    logits_rgb = out_rgb.numpy()[0].tolist()

    out_bgr = model(t_bgr)
    probs_bgr = torch.softmax(out_bgr, 1).numpy()[0].tolist()
    logits_bgr = out_bgr.numpy()[0].tolist()

print("---- RGB input ----")
print("Logits:", logits_rgb)
print("Probs:", probs_rgb)
print("Top label (RGB):", CLASS_LABELS[int(torch.argmax(torch.tensor(probs_rgb)).item())])

print("---- BGR input (channels swapped) ----")
print("Logits:", logits_bgr)
print("Probs:", probs_bgr)
print("Top label (BGR):", CLASS_LABELS[int(torch.argmax(torch.tensor(probs_bgr)).item())])

# Extra diagnostics: show first 30 checkpoint keys and first 30 model param names
print("\n---- Checkpoint keys (first 30) ----")
# show keys of the actual state we used (new_state)
print(list(new_state.keys())[:30])

print("\n---- Model param names (first 30) ----")
print([n for n,_ in model.named_parameters()][:30])

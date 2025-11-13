# local_inference_debug.py
import sys, io, math, os
import torch, timm, re
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
import numpy as np

CKPT = "best_model.pth"
MODEL_NAME = "efficientnet_lite0"
IMG_SIZE = 80  # set to the size your server uses (224 by default)
DEFAULT_TEST = "models/last_recv.jpg"

def load_checkpoint_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if "model_state" in ckpt:
        return ckpt["model_state"], ckpt
    if "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt, ckpt
    raise RuntimeError("No model_state found in checkpoint")

def clean_key(k):
    for p in ("module.", "model.", "backbone."):
        if k.startswith(p):
            k = k[len(p):]
    return k

def try_name_variants(k):
    variants = [k, clean_key(k)]
    v = re.sub(r"\.+", ".", k)
    variants.append(v)
    if v.startswith("features."):
        variants.append(v[len("features."):])
    if v.startswith("features"):
        variants.append(v[len("features"):].lstrip("."))
    return list(dict.fromkeys(variants))

def smart_assign(ck_state, model_state):
    assigned = {}
    used_model = set()
    by_shape = defaultdict(list)
    for mk, mv in model_state.items():
        by_shape[tuple(mv.shape)].append(mk)

    # exact and cleaned matches
    for ck, cv in ck_state.items():
        if ck in model_state and tuple(model_state[ck].shape) == tuple(cv.shape):
            assigned[ck] = cv; used_model.add(ck); continue
        ck_clean = clean_key(ck)
        if ck_clean in model_state and tuple(model_state[ck_clean].shape) == tuple(cv.shape):
            assigned[ck_clean] = cv; used_model.add(ck_clean); continue

    # try variants
    for ck, cv in ck_state.items():
        if any(mk for mk in used_model if mk in ck):
            continue
        for v in try_name_variants(ck):
            if v in model_state and tuple(model_state[v].shape) == tuple(cv.shape) and v not in used_model:
                assigned[v] = cv; used_model.add(v); break

    # shape-only fallback
    for ck, cv in ck_state.items():
        if any(cv is v for v in assigned.values()):
            continue
        shape = tuple(cv.shape)
        cands = [mk for mk in by_shape.get(shape, []) if mk not in used_model]
        if not cands:
            continue
        ck_tokens = re.split(r"[._]", clean_key(ck))
        def score(mk):
            return sum(1 for t in ck_tokens if t and t in mk)
        cands.sort(key=lambda m:(-score(m), m))
        pick = cands[0]
        assigned[pick] = cv; used_model.add(pick)
    return assigned

def build_model_and_map(path, num_classes=2):
    ck_state, ck_full = load_checkpoint_state(path)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    model_state = model.state_dict()
    assigned = smart_assign(ck_state, model_state)
    new_state = {k: assigned.get(k, v) for k,v in model_state.items()}
    res = model.load_state_dict(new_state, strict=False)
    print("load_state_dict result:", res)
    print("matched tensors:", len(assigned), "/", len(ck_state))
    return model

def preprocess_image(path, size=IMG_SIZE):
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(path).convert("RGB")
    t = tf(img).unsqueeze(0)  # [1,3,H,W]
    return img, t

def tensor_stats(t):
    a = t.detach().cpu().numpy()
    return {
        "shape": a.shape,
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
        "any_nan": bool(np.isnan(a).any()),
        "any_inf": bool(np.isinf(a).any())
    }

def inspect_forward(model, tensor):
    model.eval()
    with torch.no_grad():
        out = model(tensor)
    out_cpu = out.cpu().detach()
    try:
        arr = out_cpu.numpy()
        print("output numpy shape:", arr.shape)
        print("output stats: min", np.nanmin(arr), "max", np.nanmax(arr), "mean", np.nanmean(arr))
        print("any nan:", np.isnan(arr).any(), "any inf:", np.isinf(arr).any())
        flat = np.array(arr).reshape(-1)
        print("raw values (flattened first 40):", flat[:40].tolist())
        # postprocess attempt
        if flat.size == 1:
            logit = float(flat[0])
            prob = 1.0 / (1.0 + math.exp(-logit)) if math.isfinite(logit) else float('nan')
            print("single-logit -> logit:", logit, "prob:", prob)
        else:
            exps = np.exp(flat - flat.max())
            s = exps.sum()
            print("softmax sum finite:", np.isfinite(s), "sum:", s)
            if s==0 or not np.isfinite(s):
                print("softmax unstable; top raw values:", flat[:10].tolist())
            else:
                probs = exps / s
                print("top probs (first 10):", probs[:10].tolist())
    except Exception as e:
        print("could not convert output to numpy:", repr(e))

if __name__ == "__main__":
    # choose image path
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEST
    if not os.path.exists(img_path):
        print(f"Test image not found: {img_path}")
        print("Place a test image at:", DEFAULT_TEST, "or run with an explicit path.")
        sys.exit(2)
    print("Using image:", img_path)
    print("Building model with num_classes=2")
    model2 = build_model_and_map(CKPT, num_classes=2)
    img, tensor = preprocess_image(img_path, IMG_SIZE)
    print("input image size:", img.size)
    s = tensor_stats(tensor)
    print("input tensor stats:", s)
    print("Running forward (num_classes=2)...")
    inspect_forward(model2, tensor)

    print("\nBuilding model with num_classes=1 (single-logit) and testing")
    model1 = build_model_and_map(CKPT, num_classes=1)
    inspect_forward(model1, tensor)

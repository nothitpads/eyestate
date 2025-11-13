import base64
import io, argparse, time
from PIL import Image
import torch
import torchvision.transforms as T
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

IMAGE_SIZE = 80
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
CLASSES = ["open", "close"]
DEVICE = torch.device("cpu")

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

app = Flask(__name__)
CORS(app)

def import_build_model():
    """
    Try to import build_model from train.py (preferred).
    If train module is not found, raise a clear error.
    """
    try:
        from train import build_model
        return build_model
    except Exception as e:
        raise ImportError(
            "Could not import build_model from train.py. "
            "Ensure training/train.py exists and project root is in PYTHONPATH. "
            "Run from project root and set PYTHONPATH=. or copy build_model into this file."
        ) from e

def load_model_state(pth):
    ckpt = torch.load(pth, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)
    build_model = import_build_model()
    model = build_model(num_classes=len(CLASSES), cbam=True, pretrained=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    if "image" in request.files:
        img = Image.open(request.files["image"].stream).convert("RGB")
    else:
        j = request.get_json(force=True)
        b64 = j.get("image", "")
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    x = transform(img).unsqueeze(0).to(DEVICE)  # 1,C,H,W
    with torch.no_grad():
        out = model(x)
        if isinstance(out, torch.Tensor):
            probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
            pred = int(out.argmax(dim=1).item())
        else:
            logits = out[0] if isinstance(out, (list,tuple)) else out
            logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
            pred = int(logits.argmax().item())

    return jsonify({
        "pred": pred,
        "label": CLASSES[pred],
        "probs": probs,
        "latency_ms": round((time.time()-t0)*1000, 1)
    })

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="./models/best_model.pth")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model file not found: {args.model}")

    model = load_model_state(args.model)
    print("[inference] Model loaded from", args.model)
    app.run(host=args.host, port=args.port)

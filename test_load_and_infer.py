# test_load_and_infer_fixed.py
import torch
from PIL import Image
import torchvision.transforms as T
from training.train import build_model

# load model
model = build_model(num_classes=2, cbam=True, pretrained=True)
ckpt = torch.load("best_model.pth", map_location="cpu")
state = ckpt.get("model_state", ckpt)
# if checkpoint used 'module.' prefix you handled earlier; assume state matches
model.load_state_dict(state)
model.eval()

# preprocessing: match training transforms
tf = T.Compose([
    T.Resize((80,80)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

img = Image.open("./models/last_recv.jpg").convert("RGB")
x = tf(img).unsqueeze(0)  # shape [1,3,H,W]

with torch.no_grad():
    out = model(x)                     # out: torch.Tensor, shape depends on model
    out = out.detach()
    # canonicalize shape: remove batch dim if present
    if out.dim() == 2 and out.size(0) == 1:
        out = out.squeeze(0)           # e.g. [1,2] -> [2]
    # now out can be:
    #  - scalar tensor (binary-logit) -> use sigmoid
    #  - 1D tensor with 2 elements (logits for two classes) -> use softmax
    if out.numel() == 1:
        logit = out.item()             # single logit
        prob_open = 1.0 / (1.0 + float(torch.exp(-torch.tensor(logit))))
        print("Model output: single logit")
        print("logit:", logit, "prob(open):", prob_open)
    elif out.numel() == 2:
        logits = out.cpu().numpy()     # two logits, order depends on training (check your label ordering)
        # convert logits -> probabilities
        import numpy as np
        exps = np.exp(logits - np.max(logits))
        probs = exps / exps.sum()
        print("Model output: two-class logits")
        print("logits:", logits, "probs:", probs, "(probabilities in same order as training labels)")
    else:
        # unexpected shape: show diagnostics
        print("Unexpected model output shape:", tuple(out.shape))
        print(out)

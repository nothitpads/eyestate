# reexport_singlefile_onnx_fixed.py
import torch
from training.train import build_model   # exact factory from your repo
import os

# instantiate model exactly as during training
model = build_model(num_classes=2, cbam=True, pretrained=True)
ckpt = torch.load("best_model.pth", map_location="cpu")
state = ckpt.get("model_state", ckpt)
model.load_state_dict(state)
model.eval()

H, W = 80, 80           # must match training input
dummy = torch.randn(1,3,H,W)

outname = "eye_state.onnx"   # will overwrite existing file in served folder

torch.onnx.export(
    model,
    dummy,
    outname,
    input_names=["input"],
    output_names=["logits"],
    opset_version=18,               # use modern opset to avoid conversion fallback
    do_constant_folding=True,
    export_params=True,             # embed params into the file
    keep_initializers_as_inputs=False,
    dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}},
)

print("Wrote", outname, " (opset 18, export_params=True).")
print("File size (MB):", os.path.getsize(outname)/1024/1024)

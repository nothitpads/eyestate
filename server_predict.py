# server_predict.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io, math, torch, torchvision.transforms as T
from training.train import build_model

app = Flask(__name__)
CORS(app)

model = build_model(num_classes=2, cbam=True, pretrained=True)
ckpt = torch.load("best_model.pth", map_location="cpu")
state = ckpt.get("model_state", ckpt)
model.load_state_dict(state); model.eval()

tf = T.Compose([T.Resize((80,80)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files.get('file')
    if not f: return jsonify({'error':'no file'}), 400
    img = Image.open(f.stream).convert('RGB')
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x).squeeze().cpu()
        if out.numel() == 1:
            logit = float(out.item()); prob = 1/(1+math.exp(-logit)); predIdx = int(prob>0.5)
        else:
            import numpy as np
            logits = out.numpy()
            exps = np.exp(logits - logits.max()); probs = exps / exps.sum()
            predIdx = int(probs.argmax()); prob = float(probs[predIdx])
    labelMap = {0:'OPEN',1:'CLOSED'}
    return jsonify({'state': labelMap[predIdx], 'score': prob})

@app.route('/eye_state.onnx')
def onnx_route():
    return send_from_directory('.', 'eye_state.onnx')

if __name__=='__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)

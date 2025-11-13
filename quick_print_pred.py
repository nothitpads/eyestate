# quick_print_pred.py
import numpy as np

logits = np.array([30.753542, -37.861137])
exps = np.exp(logits - logits.max())
probs = exps / exps.sum()
pred_idx = int(np.argmax(probs))
pred_prob = float(probs[pred_idx])
print("pred_idx:", pred_idx, "prob:", pred_prob)
# interpret using your label map: e.g. LABELS = {0:'OPEN', 1:'CLOSED'}

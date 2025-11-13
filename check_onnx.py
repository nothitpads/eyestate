# check_onnx_external.py
import onnx, sys
fn = "eye_state.onnx"   # adjust if different
try:
    m = onnx.load(fn, load_external_data=False)
    print("OK: single-file ONNX (no external data).")
    print("Initializers:", len(m.graph.initializer))
except Exception as e:
    print("NOT single-file ONNX or load failed:", e)
    sys.exit(2)

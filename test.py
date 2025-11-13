import onnx,sys
fn = "eye_state.onnx"
try:
    m = onnx.load(fn, load_external_data=False)
    print("OK: single-file ONNX (no external data).")
except Exception as e:
    print("NOT single-file ONNX:", e)
    sys.exit(1)
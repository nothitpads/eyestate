// static/script.js  -- paste or append to existing file
const ONNX_MODEL_URL = '/models/best_model.onnx';
const INPUT_SIZE = 80;                 // MUST match export (--input_size)
const IMAGE_MEAN = [0.485, 0.456, 0.406];
const IMAGE_STD  = [0.229, 0.224, 0.225];
const CLASS_LABELS = ['CLOSED','OPEN']; // change if labels order differs

let ortSession = null;
async function initOnnxRuntime() {
  ort.env.wasm.wasmPaths = {
    // optional: leave empty to use CDN packaged wasm inside ort.min.js
  };
  // prefer wasm for broad compatibility
  ortSession = await ort.InferenceSession.create(ONNX_MODEL_URL, { executionProviders: ['wasm'] });
  console.log('ONNX loaded:', ONNX_MODEL_URL);
}
initOnnxRuntime();

/* Crop eye ROI from video using landmark array (normalized coordinates [0..1]).
   landmarks: array of 468 objects {x,y,z} as from MediaPipe FaceMesh normalized.
   indices: array of ints for the eye contour points (example sets below).
   video: HTMLVideoElement; overlay: canvas element for drawing (same dimensions). */
function cropEyeToCanvas(landmarks, indices, video, overlay) {
  if (!landmarks) return null;
  const w = overlay.width, h = overlay.height;
  const pts = indices.map(i => ({ x: landmarks[i].x * w, y: landmarks[i].y * h }));
  let minX = Math.min(...pts.map(p=>p.x)), maxX = Math.max(...pts.map(p=>p.x));
  let minY = Math.min(...pts.map(p=>p.y)), maxY = Math.max(...pts.map(p=>p.y));
  const size = Math.max(maxX - minX, maxY - minY);
  const pad = size * 0.5;
  minX = Math.max(0, Math.floor(minX - pad)); minY = Math.max(0, Math.floor(minY - pad));
  maxX = Math.min(w, Math.ceil(maxX + pad)); maxY = Math.min(h, Math.ceil(maxY + pad));
  const cw = maxX - minX, ch = maxY - minY;
  if (cw <= 0 || ch <= 0) return null;

  // scale coordinates to video pixel space (video may be different resolution than canvas)
  const sx = minX * (video.videoWidth / overlay.width);
  const sy = minY * (video.videoHeight / overlay.height);
  const sw = cw * (video.videoWidth / overlay.width);
  const sh = ch * (video.videoHeight / overlay.height);

  const off = document.createElement('canvas'); off.width = INPUT_SIZE; off.height = INPUT_SIZE;
  const ctx = off.getContext('2d');
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, INPUT_SIZE, INPUT_SIZE);
  return off;
}

function imageToTensorNCHW(canvas) {
  const ctx = canvas.getContext('2d');
  const {width, height} = canvas;
  const img = ctx.getImageData(0,0,width,height).data; // RGBA
  const floats = new Float32Array(3 * width * height);
  const wh = width * height;
  let idx = 0;
  for (let i=0; i < img.length; i += 4, idx++){
    const r = img[i] / 255.0, g = img[i+1] / 255.0, b = img[i+2] / 255.0;
    floats[idx] = (r - IMAGE_MEAN[0]) / IMAGE_STD[0];                         // R channel plane
    floats[wh + idx] = (g - IMAGE_MEAN[1]) / IMAGE_STD[1];                    // G channel plane
    floats[2*wh + idx] = (b - IMAGE_MEAN[2]) / IMAGE_STD[2];                  // B channel plane
  }
  return floats;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const ex = arr.map(a => Math.exp(a - max));
  const s = ex.reduce((a,b)=>a+b,0);
  return ex.map(v => v / s);
}

async function predictFromCanvas(canvas) {
  if (!ortSession) return null;
  const tensorData = imageToTensorNCHW(canvas);
  const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const feeds = { input: inputTensor }; // input name must match exported ONNX input
  const output = await ortSession.run(feeds);
  const outName = Object.keys(output)[0];
  const raw = output[outName].data;
  const probs = softmax(Array.from(raw));
  const topIdx = probs.indexOf(Math.max(...probs));
  return { label: CLASS_LABELS[topIdx], score: probs[topIdx], probs };
}

/* Integration: call this function inside your MediaPipe onResults callback.
   Arguments: results (MediaPipe results), video element, overlay canvas element,
   eye index arrays (leftIndices, rightIndices), and a DOM element to show state. */
async function onResults_withOnnx(results, video, overlay, leftIndices, rightIndices, stateEl) {
  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) return;
  const lm = results.multiFaceLandmarks[0]; // normalized landmarks
  const leftCanvas = cropEyeToCanvas(lm, leftIndices, video, overlay);
  const rightCanvas = cropEyeToCanvas(lm, rightIndices, video, overlay);

  const leftPred = leftCanvas ? await predictFromCanvas(leftCanvas) : null;
  const rightPred = rightCanvas ? await predictFromCanvas(rightCanvas) : null;

  let final;
  if (leftPred && rightPred) {
    const openScore = (leftPred.probs[1] + rightPred.probs[1]) / 2; // index 1 = OPEN if CLASS_LABELS aligns
    final = { label: openScore > 0.5 ? 'OPEN' : 'CLOSED', score: openScore };
  } else if (leftPred || rightPred) {
    const p = leftPred || rightPred;
    final = { label: p.label, score: p.score };
  } else {
    final = null;
  }

  if (final) {
    stateEl.textContent = `${final.label} ${ (final.score).toFixed(2) }`;
  }
}

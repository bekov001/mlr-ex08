# REVIEW — Task 2: the leaf classifier page

Page: https://bekov001.github.io/mlr-ex08/

The agent wrote `index.html`; I checked it against the four requirements below.

**Data:** I missed lab 8 (excused medical absence, ankle injury), so I do not have my group's
photo ZIP. The model is the Task 2 notebook pipeline (pretrained MobileNetV3-Small, whole network
fine-tuned, 15 epochs, Adam 1e-3) trained on the course's published photographs
`ex08_leaves_images.zip` (700 train / 120 val, 96 × 96). Final validation accuracy: **0.867**
(best epoch 0.917). Biggest confusion in the confusion matrix: cherry ↔ apple. Both have oval,
serrated leaves of similar shape and colour, so at 96 × 96 px the network mostly has the outline
and texture to go on.

## 1. It runs on the phone

The page is served over HTTPS from GitHub Pages (the camera API needs HTTPS). “Start camera” asks for
permission and opens the rear camera (`facingMode: environment`), and the probabilities update live.
The video is `playsinline muted` and the camera starts only from a tap, which is what Safari on iPhone requires.
If there is no camera, “Upload photo” classifies a picture from the gallery instead.

Phone screenshot (leaf on another screen, probabilities visible): `phone_screenshot.jpg`, added after testing on my phone.

## 2. The same numbers as the notebook

“Run check” on the page loads `check_image.jpg` and compares it with `check_expected.json`:

| class | page | notebook (`check_expected.json`) | \|diff\| |
| --- | --- | --- | --- |
| apple | 0.9998 | 0.9998 | 0.0000 |
| cherry | 0.0002 | 0.0002 | 0.0000 |
| chestnut | 0.0000 | 0.0000 | 0.0000 |
| maple | 0.0000 | 0.0000 | 0.0000 |

They agree to three decimals (checked in Chrome). I also ran the
`.onnx` file with onnxruntime in Python on the same image: same values.

## 3. The preprocessing matches the model

`model_info.json`: `input_size 96`, ImageNet `mean [0.485, 0.456, 0.406]`, `std [0.229, 0.224, 0.225]`,
the same as the notebook `transform` (`ToTensor` + `Normalize`).

In `index.html`, `toTensor()`:
- line 122–127: **central square crop.** `side = min(width, height)`, offset `(w − side)/2`, `(h − side)/2`.
  `drawImage` draws it resized to 96 × 96. The stage on screen is a square with `object-fit: cover`,
  so the visible square is exactly the region that is classified.
- line 133: **/ 255, then `(x − mean[c]) / std[c]` per channel**, the same as `ToTensor` + `Normalize`.
- line 133/136: **channel order.** The canvas gives RGBA interleaved. The loop writes channel c of pixel i to
  `data[c * plane + i]`, i.e. planar **R, G, B** (alpha dropped). The shape is `[1, 3, 96, 96]` (NCHW), which matches
  the ONNX input `image` 1×3×96×96.
- Softmax is applied in the page because the model outputs logits (no softmax in the network).

The page reads all of these values from `model_info.json`, so nothing is hard-coded. I found nothing to fix here.

## 4. Nothing leaves the phone

The page loads only:
- `ort.min.js` and its `.wasm` files from `cdn.jsdelivr.net` (ONNX Runtime Web, pinned to 1.20.1, line 87/103);
- its own files from GitHub Pages: `leaf_model.onnx`, `model_info.json`, `check_image.jpg`, `check_expected.json`, `qr.svg`.

There is no `fetch`/`XMLHttpRequest` that sends data, and no form or upload. Camera frames go only into a local canvas.
An uploaded photo is opened with `URL.createObjectURL` and decoded locally. The QR code is a static SVG
generated locally with the Python `qrcode` package (no QR web service). I decoded it with OpenCV and it gives
`https://bekov001.github.io/mlr-ex08/`.

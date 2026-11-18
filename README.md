# mlr-ex08 — Leaf classifier on the phone

MPA-MLR, exercise 8 (PyTorch). A pretrained MobileNetV3-Small fine-tuned on four leaf species
(apple, cherry, chestnut, maple), exported to ONNX and run in the phone's browser with
ONNX Runtime Web.

**Live page:** https://bekov001.github.io/mlr-ex08/

| File | What it is |
| --- | --- |
| `index.html` | the page: live rear camera (or photo upload) → central square crop → 96×96 → /255 → mean/std → ONNX model |
| `leaf_model.onnx` | the exported network (input `image` 1×3×96×96, output `logits` 1×4) |
| `model_info.json` | classes and preprocessing the page follows |
| `check_image.jpg`, `check_expected.json` | a validation image and the probabilities the notebook computed for it; the page's “Run check” button compares them |
| `qr.svg` | QR code of the page URL, generated locally |
| `REVIEW.md` | my review of the page (Task 2) |
| `REPORT.md` | my report of Task 4 |
| `task4_experiments.py`, `task4_results.json` | the Task 4 comparison (9 variants × 3 seeds) and its raw numbers |
| `best_leaf_model.onnx` | the Task 4 model I would put on the phone (MobileNet fine-tuned + augmentation) |

Data note: I missed lab 8 (excused medical absence), so I did not get my group's photo ZIP.
The model is trained on the course's published photographs of the same species
(`ex08_leaves_images.zip` in `tomasvicar/MLR-public`, 175 train + 30 val per class).

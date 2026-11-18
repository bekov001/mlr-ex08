# REPORT — Task 4: a better leaf classifier

## Setup

- **Data:** `ex08_leaves_images.zip` from the course repo (700 train / 120 val, 96 × 96, four species).
  I missed lab 8 (excused absence), so there is no group ZIP. These photographs are my only data, so
  I could not use them as “extra data” the way the assignment suggests.
- **Baseline (A):** my Task 3 `LeafNet` (4 × conv-ReLU-maxpool, 16/32/64/64 ch, FC 64, 4 outputs),
  no augmentation, Adam lr 1e-3, batch 32, 15 epochs.
- **Fair comparison:** the same training images, one change at a time against A, then combinations.
  **Every row is run with 3 seeds (0, 1, 2)**: mean ± std. The agent wrote the code in
  `task4_experiments.py` on top of my Task 3 code (same Dataset, network and loop, plus switches).
- **Two measurements:**
  - *val*: the 120 validation images, untouched;
  - *camera*: the same 120 images, each randomly cropped (50–90 %), rotated (±30°), colour-jittered and
    slightly blurred. They are generated **once** with seed 1234 and are identical for every variant.
    This set simulates a phone frame that is not centred, is tilted and has different light.
- **Noise level:** one validation image = 1/120 = **0.008** accuracy. Seed-to-seed std is 0.01–0.08,
  so differences below ~0.05 between single rows are not reliable.

## Results

| | variant | change against the row it builds on | val | camera |
| --- | --- | --- | --- | --- |
| A | baseline LeafNet | — | 0.789 ± 0.032 | 0.567 ± 0.020 |
| B | + geometric augmentation | crop 60–100 %, h/v flip, rotation 20° | 0.808 ± 0.049 | 0.625 ± 0.041 |
| C | + batch norm | BatchNorm2d after every conv | 0.814 ± 0.034 | 0.569 ± 0.046 |
| D | aug + BN (my Task 3 result) | B + C | 0.806 ± 0.081 | 0.650 ± 0.118 |
| E | D + colour jitter | + ColorJitter(0.3, 0.3, 0.3, 0.05) | 0.847 ± 0.040 | 0.728 ± 0.070 |
| F | E + regularisation, longer | dropout 0.3, AdamW wd 1e-4, label smoothing 0.1, cosine LR, 30 ep, best epoch | 0.867 ± 0.007 | 0.747 ± 0.026 |
| G | MobileNetV3-S pretrained, frozen | only the new last layer trained | 0.778 ± 0.004 | 0.653 ± 0.008 |
| H | MobileNetV3-S pretrained, fine-tuned | whole network (= Task 2 model) | 0.906 ± 0.027 | 0.814 ± 0.039 |
| **I** | **H + geometric + colour aug, cosine LR, 25 ep** | | **0.903 ± 0.010** | **0.908 ± 0.012** |

## What each change brought

- **Geometric augmentation (A → B):** +0.06 on camera, +0.02 on val. The val gain is inside the noise.
  The camera gain is about 1.5× the std, so it is probably real. That makes sense, because the camera set is
  exactly the kind of image that augmentation trains for.
- **Batch norm (A → C):** +0.025 val, **nothing on camera**. Training converges faster, but on its own it does
  not make the network more robust. Not larger than the noise.
- **Aug + BN (D):** better on camera than either alone, but very unstable between seeds (std 0.08 / 0.12).
  One seed collapsed to 0.69 val. In my single-seed Task 3 run, D looked like the clear winner (0.883).
  With 3 seeds it is no better than B on val, which shows why one run is not enough.
- **Colour jitter (D → E):** +0.08 on camera, +0.04 on val. This is the largest single gain for the small network.
  Phone photos differ mostly in light and white balance, which colour jitter imitates.
- **Regularisation + longer training + best epoch (E → F):** small mean gain, but the seed std drops from
  0.04 to 0.007. Its main effect is stability, not accuracy.
- **Pretrained, frozen (G):** worse than my small network trained from scratch on val (0.78). ImageNet
  features alone do not separate apple from cherry, and the network has to adapt.
- **Pretrained, fine-tuned (H):** +0.12 val and +0.25 camera over the baseline, by far the biggest jump.
  The pretrained features beat everything I did with the small network.
- **Fine-tuned + augmentation + cosine (H → I):** val unchanged (0.906 vs 0.903, noise), but **camera
  0.814 → 0.908** and much lower std. The gain shows only on the camera-like set. Judged by val alone,
  I would have missed it.

## What did not help

- Batch norm alone on camera-like images (C ≈ A).
- Freezing the pretrained network (G): worse than fine-tuning, and on val even worse than the small network.
- On the clean val set, augmentation for the pretrained model (H vs I): equal within noise.

## Errors

Confusion matrix of I (seed 0) on val: apple → cherry 5, cherry → apple 4, maple → cherry 1,
chestnut 30/30. Apple ↔ cherry stays the main confusion in every variant (the Task 2 model also has
5 + 5 of these errors). The leaves have a similar oval, serrated shape, and at 96 px the fine vein and edge
detail is mostly lost.

## Which model I would put on the phone

**I: pretrained MobileNetV3-Small, fully fine-tuned with geometric + colour augmentation and a cosine
LR schedule, 25 epochs.** It has the same val accuracy as the Task 2 model, but it is clearly the best and most
stable on camera-like images (0.908 ± 0.012), and the phone camera is where it will be used.
It is exported as `best_leaf_model.onnx` (same input/preprocessing as `model_info.json`) and can be
loaded in the course camera app. The page `index.html` keeps the Task 2 model `leaf_model.onnx`,
because the self-check against `check_expected.json` belongs to that model.

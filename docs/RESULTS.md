# Ultrasound experiment results

**Study:** Explainable Transfer Learning Framework for Fetal Ultrasound Standard Plane Classification.

All values below originate from the executed study and its saved validation/final-test records. This document does **not** present the held-out TEST as an additional model-selection experiment.

## Model selection on VALIDATION only

Both frozen-backbone models used the same patient-separated **1,311-image validation split**.

| Metric | U1 MobileNetV2 | U2 EfficientNetB0 |
|---|---:|---:|
| Validation accuracy | **88.10%** | 86.8802% |
| Validation macro-F1 | **0.8572** | 0.8507 |
| Selected epoch (minimum validation loss) | 2 | 5 |
| Minimum validation loss | 0.3552 | **0.3521** |

U1 was chosen for final evaluation because of its higher validation macro-F1 and accuracy, supported by per-class results. The selection criteria were discussed based on observed validation findings, and should not be described as preregistered. U2 showed higher fetal-abdomen F1 (**0.7750** vs U1's **0.7229**) but lower recall for `Other` (**0.6325** vs U1's **0.7778**). Neither was universally better on every class or metric.

## One-time final held-out TEST, U1 only

The frozen U1 checkpoint was evaluated **once** on **5,271 images from 896 held-out patient groups**.

| U1 TEST metric | Measured result |
|---|---:|
| Accuracy | **85.4297%** |
| Macro-F1 | **0.834528** |
| Macro-precision | **0.818983** |
| Macro-recall | **0.860314** |

**No final TEST result exists for U2.** The test was not used to retune or choose between U1 and U2. Precise per-class TEST values, raw predictions and the saved test confusion matrix should be taken from the original saved Drive evidence, not reconstructed or guessed from these aggregate metrics.

## Grad-CAM on VALIDATION examples

The frozen U1 MobileNetV2 produced Grad-CAM overlays for **12 preselected validation examples** (two per class, 12 distinct patient groups), targeting backbone layer `out_relu`. Saved-model inference matched historical predictions on **12/12** selected examples; **10/12** happened to be predicted correctly. This is descriptive only, **not** a new TEST score or a measure of general explanation quality.

See `notebooks/U_GradCAM_U1_MobileNetV2.ipynb` and `docs/GRADCAM_IMAGE_PROVENANCE.md` for example attribution and visualization provenance.

## Interpretation limits

The held-out TEST is patient-separated within the original dataset. It is not an independent external-site study. Differences in validation and TEST scores are descriptive results, not evidence of clinical safety or calibrated probability. Grad-CAM highlights regions influencing the model's computation; those overlays do not establish medical correctness.

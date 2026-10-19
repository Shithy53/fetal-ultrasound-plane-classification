# U1 Grad-CAM image provenance

This repository stores an unmodified executed copy of
`notebooks/U_GradCAM_U1_MobileNetV2.ipynb` and an image exported byte-for-byte
from the notebook's Section 6 displayed output.

- Figure: `results/figures/U1_GradCAM_validation_example_01.png`
- Source dataset: FETAL_PLANES_DB maternal-fetal ultrasound image collection.
- Source partition: **VALIDATION**, never the held-out TEST partition.
- Figure content: source ultrasound input, predicted-class Grad-CAM heatmap,
  and alpha-blended overlay from the frozen U1 MobileNetV2 model.
- Example: fixed validation example 01, true and predicted label `Fetal abdomen`.
- Explanation implementation: MobileNetV2 `out_relu` feature activation,
  attribution for the model's predicted class.
- The notebook's 12 examples were selected before evaluating predictions,
  two per class and across 12 separate patient groups. All 8 code sections
  retain their historical saved outputs.
- This figure is not independent diagnostic or anatomical-localization evidence.
- Dataset attribution and any reuse terms should follow the original
  FETAL_PLANES_DB source. Repository access is currently intended to remain Private.

Provenance checks:

- Original executed notebook SHA-256: `be3fd19fc6d6ef70f9b80bfdc55b7d4faa90c4d67057e55b68afd2cf72af5ac5`
- Extracted displayed PNG SHA-256: `31afbc2dc37cf9175565b851f658531c97d0c75e6ccab5efd5e3b074da7b5f8e`
- PNG output location in original notebook JSON: cell index 12, output index 0
- Notebook source code, metadata and all historical outputs are unchanged.
- No model training or held-out TEST evaluation was run to prepare this package.

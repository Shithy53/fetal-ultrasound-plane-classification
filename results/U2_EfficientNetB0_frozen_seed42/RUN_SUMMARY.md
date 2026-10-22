# U2_EfficientNetB0_frozen_seed42 - measured experiment summary

## Scope
A frozen ImageNet EfficientNetB0 backbone with a six-class ultrasound classification head.
No fine-tuning, additional training augmentation, Grad-CAM, or held-out test evaluation.

## Dataset and protocol
FETAL_PLANES_DB: 5,818 training images,
1,311 validation images,
5,271 held-out test images (unused).
The exact saved patient-disjoint split is reused, with 180
validation patient IDs. No new split was generated.
Class order: ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other'].
Images were resized with padding to 224 x 224 and
passed as float32 in [0, 255]. EfficientNetB0 includes its own preprocessing.
Batch size 16, seed 42, Adam learning rate 0.001,
head dropout 0.2, balanced training class weights and no extra augmentation.
Training cross-entropy uses class weights; validation loss is unweighted.

## Validation results (measured U2)
- Completed epochs: 6 of 6 maximum.
- Best checkpoint selected by minimum val_loss: epoch 5.
- Validation loss: 0.352122.
- Validation accuracy: 0.868802.
- Validation macro-F1: 0.850739.
- Validation weighted-F1: 0.867353.

For reference, the handoff reports U1 validation accuracy
0.8810 and macro-F1 0.8572
on the same manifest. These U1 numbers were not recalculated by U2.
Confusion-matrix rows are true labels and columns are predicted labels.
Per-class precision/recall/F1 and per-image scores are exported separately.

## Interpretation and limitations
These are validation results, not held-out test performance or external clinical
validation. Class frequency and category-specific errors require consideration.
Softmax scores are uncalibrated model outputs, not medical confidence estimates.
The CSVLogger epoch column is zero-based; the report and figures use one-based epochs.

## Provenance
Checkpoint SHA-256: b13d3601e03a34b452b4af284b1fbfad316790fe6d8be4c7d69aaf1537e8280a
Manifest SHA-256: ef6fec1a70e80a0b7ed34c990ebac2918ddc8a81dcef300aa244ce6ab1a5e1e0
Archive MD5: 2a5fcc2cefb789bcc0f6c1f73e0ea43f
Notebook revision: 2026-10-19-u2-v1

The executed notebook is saved separately from the evidence ZIP.

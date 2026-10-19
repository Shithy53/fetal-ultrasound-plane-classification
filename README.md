# Explainable Fetal Ultrasound Standard Plane Classification

Undergraduate research project on classifying six categories of maternal-fetal ultrasound images using transfer learning and visual explanations.

**Submitted By:** Kazi Nusrat Jahan Shithy  
**Supervisor:** Prof. Dr. Md. Zahidul Islam  
**Department:** Dept. of Information and Communication Technology, Islamic University, Kushtia, Bangladesh

> **Research and educational use only.** This is standard-plane image classification, not fetal disease diagnosis. The models have not been clinically or externally validated. Softmax outputs should not be interpreted as calibrated medical probabilities.

## Research objectives

- Compare frozen ImageNet-pretrained MobileNetV2 and EfficientNetB0 feature extractors for six-class ultrasound plane classification.
- Preserve patient-separated training, validation, and held-out testing.
- Provide Grad-CAM visualizations from the selected model and examine classification errors.
- Document experiments and one-time final evaluation with traceable saved artifacts.

## Dataset and partitioning

The project uses **FETAL_PLANES_DB** with **12,400 images** and **1,792 patient IDs**. Existing audited partitions were reused unchanged.

| Partition | Images | Patient groups |
|---|---:|---:|
| Training | 5,818 | 716 |
| Validation | 1,311 | 180 |
| Held-out TEST | 5,271 | 896 |

The original official TEST partition was preserved, and patient groups do not overlap between partitions. The fixed class mapping is:

| ID | Category |
|---:|---|
| 0 | Fetal abdomen |
| 1 | Fetal brain |
| 2 | Fetal femur |
| 3 | Fetal thorax |
| 4 | Maternal cervix |
| 5 | Other |

The dataset is **not included in this repository**. Obtain it from its original authorized source and follow the source's data-use and licensing requirements. No new split should be generated to reproduce the recorded study.

## Model comparison (VALIDATION only)

| Experiment | Frozen backbone | Validation accuracy | Validation macro-F1 |
|---|---|---:|---:|
| U1 | MobileNetV2 | 88.10% | 0.8572 |
| U2 | EfficientNetB0 | 86.8802% | 0.8507 |

**U1 MobileNetV2** was selected based primarily on validation macro-F1, supported by accuracy and per-class analysis. Individual model checkpoints were chosen by minimum validation loss. This selection was finalized and recorded before the held-out TEST was accessed.

For U1, images are resized with padding to **224 x 224 x 3**, and MobileNetV2 input rescaling is contained within the saved model. U2 uses EfficientNetB0's native model preprocessing, not U1's MobileNetV2 scaling.

## Final held-out TEST (selected U1 only)

The frozen U1 model was evaluated once on **5,271 patient-disjoint TEST images**.

| TEST metric | Result |
|---|---:|
| Accuracy | **85.4297%** |
| Macro-F1 | **0.834528** |
| Macro-precision | **0.818983** |
| Macro-recall | **0.860314** |

U2 was **not** evaluated on the held-out TEST. Do not use TEST results for model selection or tuning, and do not rerun the one-time TEST merely to populate repository outputs. Report these TEST metrics separately from U1/U2 validation comparisons.

## Explainability

Grad-CAM was generated for the selected U1 model on **12 predefined VALIDATION images**, two from each category and from 12 distinct patient groups. The target was MobileNetV2's `out_relu` feature layer. The resulting heatmaps are illustrative model explanations, not proof of anatomically correct or causal decision-making.

## Repository notebooks

The project-specific setup utility is provided in `notebooks/00_Ultrasound_Setup_and_Data_Checks.ipynb`. It creates missing ultrasound folders, restores the verified archive when necessary and checks the existing patient-separated manifest. This new utility has no saved execution outputs and must not regenerate the recorded split.

The following four notebooks preserve the actual executed experiments, Grad-CAM outputs and final evaluation:

- `notebooks/U1_Ultrasound_MobileNetV2_Baseline.ipynb`
- `notebooks/U2_Ultrasound_EfficientNetB0_Baseline.ipynb`
- `notebooks/U_GradCAM_U1_MobileNetV2.ipynb`
- `notebooks/U_Final_Test_U1_MobileNetV2.ipynb`

Source code, original outputs and signed model-selection evidence should not be rewritten to imply another training run or TEST evaluation.

Some notebooks assume **Google Colab** and persistent paths under `/content/drive/MyDrive/ICTIU_Final_Projects/`. The saved model checkpoints, raw predictions, datasets and full evidence bundles remain outside GitHub in controlled Google Drive storage. The notebooks are historical execution records, not turnkey instructions to rerun training or held-out TEST.

**Observed project environment in later Colab sessions:** Python 3.13.15, TensorFlow 2.21.0, Keras 3.13.2, NVIDIA T4, random seed 42. Check each notebook's saved environment record before attempting a fresh implementation; exact GPU training reproducibility is not guaranteed.

## Provenance and research limitations

- Ultrasound split manifest SHA-256: `ef6fec1a70e80a0b7ed34c990ebac2918ddc8a81dcef300aa244ce6ab1a5e1e0`
- Ultrasound selected U1 checkpoint SHA-256: `5154c6d97685c2ef39d3ba040a409443cc0f261e070ca627c24b940979b897ae`
- Selection freeze record: `U1_model_selection_freeze_2026-10-19.json`
- Final-test protocol record: `U1_final_test_protocol_2026-10-19.json`

Limits include a single data source, no independent external-hospital test, limited model selection scope and no claim of clinical suitability. Repository ownership, Git commit authorship and academic project authorship are separate concepts and should be recorded accurately.

**Data and model redistribution:** Raw medical images, large checkpoint files and restricted third-party assets are intentionally excluded. Permission to redistribute any sample image or heatmap derived from the dataset must be checked against the original source terms before committing it, including inside notebook outputs.

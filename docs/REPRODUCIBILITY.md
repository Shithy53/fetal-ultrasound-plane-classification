# Reproducibility and preservation guide

## Scope of what this repository preserves

These files document the observed **executed** ultrasound experiments, the preselected validation Grad-CAM examples and the **single final U1 TEST**. The notebooks were originally executed in Google Colab with persistent Drive folders. Retained notebook outputs are historical execution evidence, not instructions to retrain or to repeat the final held-out evaluation.

Notebooks that should exist under `notebooks/`:

| Notebook | Role | Saved-output code cells |
|---|---|---:|
| `U1_Ultrasound_MobileNetV2_Baseline.ipynb` | Frozen MobileNetV2 baseline, validation | 7/7 |
| `U2_Ultrasound_EfficientNetB0_Baseline.ipynb` | Frozen EfficientNetB0 baseline, validation | 10/10 |
| `U_GradCAM_U1_MobileNetV2.ipynb` | U1 Grad-CAM on preselected validation images | 8/8 |
| `U_Final_Test_U1_MobileNetV2.ipynb` | Previously executed and finalized one-time U1 TEST | 8/8 |

The original **shared** `00_Setup_and_Data_Checks.ipynb` prepared the two separate project datasets and remains archived outside this curated ultrasound notebook sequence. A proposed ultrasound-only derived setup notebook has **not yet been authored**. Such a copy, if later approved, must reuse the fixed manifest and cite this original preparation rather than silently producing a new split.

## Recorded execution environment

- Google Colab, NVIDIA Tesla T4 (15,360 MiB), Python **3.13.15** in the U2 run.
- TensorFlow **2.21.0**, Keras **3.13.2**; seed **42**.
- The exact dependency versions in the executed U2 environment are recorded in the root `requirements.txt`. Colab-specific APIs such as `google.colab.drive` are provided by Colab.
- Input resolution for both models is **224 x 224 x 3** with resize-with-padding and float32 image pixels. **U1** includes MobileNetV2 scaling inside the saved model. **U2** uses EfficientNetB0's own model preprocessing. Do **not** apply U1's `[-1,1]` scaling to U2.
- Software, GPU and non-deterministic operations can affect a fresh retraining run. Matching package versions alone does not guarantee identical training results.

## Saved evidence locations (NOT copied to GitHub)

```text
/content/drive/MyDrive/ICTIU_Final_Projects/ultrasound/
    results/U1_MobileNetV2_frozen_seed42/
    results/U2_EfficientNetB0_frozen_seed42/
    results/U_GradCAM_U1_validation_seed42/
    results/U_Final_Test_U1_seed42/
    selection/U1_model_selection_freeze_2026-10-19.json
    selection/U1_final_test_protocol_2026-10-19.json
```

The full model checkpoints, complete prediction CSVs and saved evidence archives are retained in Drive, not distributed in GitHub. Paths above are provenance references, not proof that another user has access to those private files.

Selected checkpoint SHA-256: `5154c6d97685c2ef39d3ba040a409443cc0f261e070ca627c24b940979b897ae` (U1 `best_model.keras`). This should be checked against the actual stored file if artifact access is available. The original model-selection freeze and final-test protocol were written **before** TEST inference.

## Read-only verification workflow

1. Open the notebooks on GitHub to inspect the original code and **saved** outputs. Do not use `Run all` merely to verify their existence.
2. Compare the aggregate **validation** and **TEST** numbers with `docs/RESULTS.md`, keeping partitions distinct. U2 has no final TEST metric.
3. Where authorized Drive files are available, independently hash a saved checkpoint/manifest and compare it with the recorded fingerprints. This does not need image extraction, model training or inference.
4. Refer to the existing saved predictions/metrics and original report figures when preparing the thesis. Any additional programmatic checks should operate on **saved outputs**, not repeat final TEST inference.

A test of a newly modified pipeline is **not** the same as recreating historical execution. If future code is adapted for portability, save it separately and clearly disclose that it has not been re-executed.

## Historical notebook byte fingerprints

These SHA-256 values identify the original executed notebook files as uploaded to the project. Normal documentation edits would change notebook hashes without changing the past experimental results.

| Executed notebook | SHA-256 |
|---|---|
| U1 baseline | `4d6803aa71dcd922ba9d2216ec55d3615457561bedbc70c50ddefcebf595457c` |
| U2 baseline | `e6fcadd1a08341372710a0f3bc18cd4444d56b63527acf7d1a5027661d24334a` |
| U1 Grad-CAM | `be3fd19fc6d6ef70f9b80bfdc55b7d4faa90c4d67057e55b68afd2cf72af5ac5` |
| U1 final TEST | `e691872a44ec4ae83a0f22fa93edce3c27353e60f76e538db84b8f2a6eb49f10` |
| Original shared data setup (archived) | `2fb9624d36ff129edbc27b6ca7ec917e20e93e1f1e06b550b16884b039a158a5` |

## Safety and attribution

- Do not commit access tokens, credential files, private patient metadata, unauthorized dataset archives, or large `.keras` models.
- Treat example images and Grad-CAM overlays as source-derived visual material and credit their dataset appropriately; Private visibility does not replace applicable redistribution permissions.
- Grad-CAM visual attention is explanatory only. It does not establish causal relevance, anatomical correctness or clinical usefulness.
- Academic student attribution, repository ownership and Git commit authorship are distinct.

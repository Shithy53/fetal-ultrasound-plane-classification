# Dataset and fixed split provenance

## Source and scope

This project uses the publicly released **FETAL_PLANES_DB** fetal ultrasound image collection (Zenodo record [3904280](https://zenodo.org/records/3904280)). The recorded patient-group assignments, class mapping and audit evidence are preserved as fixed ultrasound study artifacts. The repository notebook `notebooks/00_Ultrasound_Setup_and_Data_Checks.ipynb` supports archive recovery and verification without creating another split or training a model.

The ultrasound task predicts six **standard-plane categories**, not fetal disease status or fetal brain subplane subclasses:

| Numeric ID | Class label |
|---:|---|
| 0 | Fetal abdomen |
| 1 | Fetal brain |
| 2 | Fetal femur |
| 3 | Fetal thorax |
| 4 | Maternal cervix |
| 5 | Other |

## Frozen partitioning

| Partition | Image count | Distinct patient IDs |
|---|---:|---:|
| Training | 5,818 | 716 |
| Validation | 1,311 | 180 |
| Final held-out TEST | 5,271 | 896 |
| **Total** | **12,400** | **1,792** |

The original dataset's official test patients were retained as the held-out TEST. Training and validation were separated by patient group. The saved audit records show no patient group spanning the three partitions. The same fixed partition assignments and class-ID mapping were used for U1, U2, Grad-CAM selection and the selected U1's one-time final TEST.

**Do not regenerate or reshuffle the original split.** In particular, a random image-row split would undermine patient separation. Do not use held-out TEST data for model selection, preprocessing choices, threshold setting or any additional evaluation.

## Archived input and provenance

Persistent data locations used in the executed Google Colab notebooks (these are paths within the user's own mounted Google Drive):

```text
/content/drive/MyDrive/ICTIU_Final_Projects/ultrasound/
    archives/FETAL_PLANES_ZENODO.zip
    metadata/FETAL_PLANES_DB_data.csv
    splits/manifest_candidate.csv
    splits/class_mapping.json
    results/audit.json
```

Selected verified provenance fingerprints:

| Source record | SHA-256 |
|---|---|
| Ultrasound `manifest_candidate.csv` | `ef6fec1a70e80a0b7ed34c990ebac2918ddc8a81dcef300aa244ce6ab1a5e1e0` |
| Ultrasound `class_mapping.json` | `26ec8f0a94d29d3756f36c52f02ddaad0d953403c0cd3d3903a243547d693428` |
| Frozen TEST row order | `da93cb45f9cab66a7abc498fc116f7200253d4067b3a2c59a23fdffc170060f8` |

The supplied input dataset, patient identifiers, dataset ZIP and any patient-level manifest records are **not required in GitHub** and should remain in controlled external storage. Any released image examples must comply with the source's actual permissions and attribution requirements. The repository intentionally uses aggregate results to describe partitions, not identifiable raw patient metadata.

## Limits

The held-out TEST is patient-separated **within this dataset**. It is not an external-hospital, multi-site or prospective clinical evaluation. Patient-separated split verification does not establish broader clinical generalization or calibrated confidence scores.

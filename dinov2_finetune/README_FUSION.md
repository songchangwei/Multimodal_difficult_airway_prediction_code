# Multi-image frozen-feature fusion

The fusion implementation accepts any number of image columns. For each
fold, it loads the matching already-trained single-image DINOv2 checkpoint,
extracts the backbone feature, freezes that extractor, concatenates features
in the YAML order, and trains only the fusion classifier.

To use all currently trained image models, run:

```bash
python dinov2_finetune/train_fusion_cv.py --config dinov2_finetune/configs/fusion_all_images.yaml
```

The all-image configuration combines:

- `外观检查_1`, `外观检查_2`
- `颈围、甲颏距离和胸颏距离_1`, `颈围、甲颏距离和胸颏距离_2`
- `张口度_1`, `上唇咬合试验_1`, `牙齿检查_1`, `Mallampati分级_1`

Rows missing any configured image are excluded because every sample must have
a complete set of image features.

The existing two-image configurations remain supported:

- `outputs/颈围、甲颏距离和胸颏距离_1/fold_{k}/best.pt`
- `outputs/颈围、甲颏距离和胸颏距离_2/fold_{k}/best.pt`

For each fold, it loads the corresponding checkpoints, extracts the
fine-tuned DINOv2 backbone features for both images, freezes those features,
concatenates them, and trains only a small fusion classifier. The two
checkpoints are loaded from the same fold as the fusion model to avoid using
the validation fold while building features.

Run from the repository root:

```bash
python dinov2_finetune/train_fusion_cv.py --config dinov2_finetune/configs/fusion_neck_distance.yaml
```

For the two physical-examination views, run:

```bash
python dinov2_finetune/train_fusion_cv.py --config dinov2_finetune/configs/fusion_physical_examination.yaml
```

Results are written to:

```text
outputs/颈围、甲颏距离和胸颏距离_1_颈围、甲颏距离和胸颏距离_2_frozen_fusion/
  fold_0/
  ...
  cv_summary.json
  cv_summary.csv
```

## All images plus structured data

The multimodal experiment uses all eight currently trained image columns and
the structured variables from `structured_finetune/config.yaml`:

- categorical: `性别`, `种族`, `age_group`, `bmi_group`
- numerical: `age`, `height_cm`, `weight_kg`, `BMI`, `age_squared`,
  `bmi_squared`, `age_bmi`

For every fold, the matching single-image `best.pt` checkpoints are loaded,
used to extract DINOv2 features, and frozen. The structured-data imputer,
one-hot encoder, and scaler are fitted on the training fold only. Only the
image-structured fusion head is optimized.

Run from the repository root:

```bash
python dinov2_finetune/train_multimodal_cv.py \
  --config dinov2_finetune/configs/fusion_all_images_structured.yaml
```

Results are written to:

```text
outputs/外观检查_1_外观检查_2_颈围、甲颏距离和胸颏距离_1_颈围、甲颏距离和胸颏距离_2_张口度_1_上唇咬合试验_1_牙齿检查_1_Mallampati分级_1_structured_frozen_fusion/
  fold_0/
  ...
  cv_summary.json
  cv_summary.csv
```

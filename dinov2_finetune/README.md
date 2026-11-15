# DINOv2 Fine-tuning for Difficult Airway Image Classification

This project fine-tunes a DINOv2 image backbone for binary classification.

Default task:
- input: one image per sample
- label: `0/1`
- split: use the generated `folds/fold_{k}_train.csv` and `folds/fold_{k}_val.csv`
- training mode: one model per image column
- metrics: AUC, F1, sensitivity, specificity, accuracy, precision, recall

## Directory

```text
dinov2_finetune/
  configs/
    default.yaml
  src/
    config.py
    data.py
    logger.py
    metrics.py
    model.py
    train.py
  train_cv.py
```

## Install

```bash
pip install -r requirements.txt
```

## Train 5 folds

Set `model_name` in `configs/default.yaml` to your DINOv2 Hugging Face model id or local checkpoint path.

```bash
python dinov2_finetune/train_cv.py --config dinov2_finetune/configs/default.yaml
```

## Outputs

```text
outputs/
  外观检查_1/
    fold_0/
      train.log
      metrics.csv
      metrics.json
      best.pt
      val_predictions.csv
    ...
  张口度_1/
    fold_0/
    ...
  cv_summary.json
  cv_summary.csv
```

## Notes

- The current pipeline trains one model for each image column listed in config.
- Video columns are ignored by default.
- Each column is trained independently across all folds.

## Training-set Grad-CAM heatmaps

After training, generate heatmaps from the training split with each fold's fine-tuned `best.pt` checkpoint:

```bash
python dinov2_finetune/visualize_heatmaps.py \
  --config dinov2_finetune/configs/default.yaml \
  --image-column "Mallampati分级_1" \
  --fold all
```

Use `--fold 0` to process one fold, and optionally add `--limit 32` for a small preview. The script uses the matching `fold_{k}_train.csv` and `outputs/<image-column>/fold_{k}/best.pt`. It writes an overlay image, a color heatmap, and `predictions.csv` under `outputs/heatmaps/<image-column>/fold_{k}/`. By default, Grad-CAM explains the predicted class; `--target label`, `--target positive`, and `--target negative` select the ground-truth, positive, or negative class instead.

# VideoMAE Fine-tuning for Difficult Airway Video Classification

This project fine-tunes a VideoMAE backbone for binary video classification.

Default task:
- input: one video per sample
- label: `0/1`
- split: use `folds/fold_{k}_train.csv` and `folds/fold_{k}_val.csv`
- training mode: one model per video column
- metrics: AUC, F1, sensitivity, specificity, accuracy, precision, recall

## Directory

```text
videomae_finetune/
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

```bash
python videomae_finetune/train_cv.py --config videomae_finetune/configs/default.yaml
```

## Outputs

```text
video_outputs/
  前屈和后仰_1/
    fold_0/
      train.log
      metrics.csv
      metrics.json
      best.pt
      val_predictions.csv
    ...
  左右旋转_1/
    fold_0/
    ...
  cv_summary.json
  cv_summary.csv
```

## Notes

- The current pipeline trains one model for each video column listed in config.
- In the current default config, only `前屈和后仰_1` and `前屈和后仰_2` are used.
- Each validation CSV stores video-level probabilities.
- The default backbone is local `videomae-base`.

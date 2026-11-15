# Two-video frozen-feature fusion

This pipeline uses the two existing single-video VideoMAE models:

- `video_outputs/前屈和后仰_1/fold_{k}/best.pt`
- `video_outputs/前屈和后仰_2/fold_{k}/best.pt`

For each fold, it loads the corresponding checkpoints, extracts the
fine-tuned VideoMAE backbone features for both videos, freezes those
features, concatenates them, and trains only a small fusion classifier.

Run from the repository root:

```bash
python videomae_finetune/train_fusion_cv.py --config videomae_finetune/configs/fusion_flexion_extension.yaml
```

Results are written to:

```text
video_outputs/前屈和后仰_1_前屈和后仰_2_frozen_fusion/
  fold_0/
  ...
  cv_summary.json
  cv_summary.csv
```

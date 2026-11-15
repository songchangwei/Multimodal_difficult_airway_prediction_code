# Structured Data Modeling

Use demographic variables to predict difficult airway:

- sex
- age
- height
- weight
- race
- BMI derived from height and weight

The project compares at least eight algorithms:

- Logistic Regression
- Random Forest
- Extra Trees
- HistGradientBoosting
- Gradient Boosting
- AdaBoost
- RBF-SVM
- KNN

Metrics:

- AUC
- F1
- accuracy
- precision
- recall
- sensitivity
- specificity
- TN / FP / FN / TP

## Run

```bash
python structured_finetune/train_cv.py --config structured_finetune/config.yaml
```

## Outputs

```text
structured_outputs/
  logistic_regression/
    fold_0/
      train.log
      metrics.json
      val_predictions.csv
    ...
  random_forest/
  hist_gradient_boosting/
  cv_summary.csv
  cv_summary.json
```

The split files must already exist:

```text
folds/fold_0_train.csv
folds/fold_0_val.csv
...
```

`医生诊断` and `困难气道` are deliberately not used as input features because they may cause label leakage.

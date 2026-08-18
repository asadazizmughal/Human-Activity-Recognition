# Human Activity Recognition from Accelerometer Data

Classifying six daily activities (walking, walking upstairs, walking downstairs,
sitting, standing, laying) from wearable inertial sensor data, and comparing
classical machine learning on engineered features against deep learning on raw
signals.

The point of this project is not to hit the highest possible accuracy. It is to
evaluate models the way they would actually be used: on people the model has
never seen before. Most public HAR tutorials split windows randomly, which leaks
the same subject into both train and test and inflates accuracy. This project
keeps the dataset's built-in subject-independent split, so every result reflects
generalisation to new users.

## Background

I worked on real clinical sensor data during a research internship at EuroMov
(Digital Health in Motion), predicting real-world arm use in stroke patients
from wrist-worn accelerometers. That work made one thing clear: with sensor
data you cannot let a subject appear in both training and test, or your numbers
lie. This project applies the same evaluation discipline to a public benchmark.

## Dataset

[UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
30 subjects wore a waist-mounted smartphone. Accelerometer and gyroscope were
recorded at 50 Hz, then segmented into 2.56 second windows (128 samples, 50%
overlap).

The data ships in two forms, and this project uses both:

| Representation      | Shape                    | Used by            |
|---------------------|--------------------------|--------------------|
| Engineered features | (n_windows, 561)         | Classical ML track |
| Raw inertial signal | (n_windows, 128, 9)      | Deep learning track|

Split (shipped with the dataset, kept intact):

| Split | Subjects | Windows |
|-------|----------|---------|
| Train | 21       | 7352    |
| Test  | 9        | 2947    |

No subject appears in both splits.

## Approach

Two tracks, and the comparison between them is the finding.

**Classical ML on engineered features**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- SVM (RBF kernel)

**Deep learning on raw signal windows**
- 1D CNN
- BiLSTM
- (optional) CNN-LSTM hybrid

Evaluation leads with macro-F1 (the classes are imbalanced), alongside accuracy,
per-class precision and recall, and a confusion matrix.

## Results

_To be filled in as models are built._

| Model               | Accuracy | Macro-F1 | Train time |
|---------------------|----------|----------|------------|
| Logistic Regression | –        | –        | –          |
| Random Forest       | –        | –        | –          |
| Gradient Boosting   | –        | –        | –          |
| SVM (RBF)           | –        | –        | –          |
| 1D CNN              | –        | –        | –          |
| BiLSTM              | –        | –        | –          |

## Repository layout

```
har-accelerometer/
├── README.md
├── requirements.txt
├── data/                    # downloaded on demand, gitignored
├── src/
│   ├── data_loader.py       # download + load features and raw signals
│   ├── preprocessing.py     # (Step 2)
│   ├── features.py          # (Step 2)
│   ├── models.py            # (Step 3-4)
│   └── evaluate.py          # (Step 3)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_models.ipynb
│   └── 04_deep_learning.ipynb
├── results/
│   └── confusion_matrices/
└── report/
    └── findings.md
```

## Getting started

```bash
pip install -r requirements.txt

# Downloads the dataset (~59 MB) on first run, then prints a data summary.
python src/data_loader.py
```

The loader handles the download, unpacks the dataset (the archive ships a
zip inside a zip), and exposes two functions:

```python
from src.data_loader import load_features, load_raw_signals

X_train, y_train, subjects_train = load_features("train")   # (7352, 561)
X_raw,   y_raw,   subjects_raw   = load_raw_signals("train") # (7352, 128, 9)
```

## Status

- [x] Step 1: data foundation (loader, both representations, verified split)
- [ ] Step 2: EDA and feature engineering
- [ ] Step 3: classical models and evaluation
- [ ] Step 4: deep learning track
- [ ] Step 5: findings write-up and README polish

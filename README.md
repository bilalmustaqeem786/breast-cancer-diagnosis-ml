# Breast Cancer Diagnosis with Machine Learning

A classification project that predicts whether a breast tumour is malignant or benign from cell-nucleus measurements, comparing Logistic Regression and Random Forest. Built as a portfolio demonstration of applied machine learning on health data.

> **Note:** This project is for learning and portfolio demonstration only. It is **not** clinical or medical advice.

## Features

- Exploratory analysis of the Wisconsin breast cancer dataset (569 samples, 30 features)
- Train/test split (80/20, stratified) with feature scaling for Logistic Regression
- Logistic Regression and Random Forest (300 trees) trained and compared side by side
- Accuracy, precision, recall, confusion matrices and classification reports
- Random Forest feature-importance ranking with a saved bar chart

## Tech Stack

- Python
- pandas, NumPy, scikit-learn, matplotlib

## Results

Test set: 114 samples (42 malignant, 72 benign).

| Model | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: |
| Logistic Regression | **0.9825** | 0.9861 | 0.9861 |
| Random Forest | 0.9474 | 0.9583 | 0.9583 |

Top 3 most important features (Random Forest): worst perimeter (0.1373), worst area (0.1373), worst concave points (0.1155).

## Project Structure

```
breast-cancer-diagnosis-ml/
├── src/
│   └── train.py          # Trains both models, prints metrics, saves the chart
├── notebooks/
│   └── 01_breast_cancer_classification_ml.ipynb
├── outputs/
│   └── feature_importance.png
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the training script
python src/train.py
```

The script prints both models' metrics and saves the feature-importance chart to `outputs/feature_importance.png`.

## Dataset

The scikit-learn built-in breast cancer dataset (`sklearn.datasets.load_breast_cancer`) — no download and no real patient records are used. Features are computed from digitised images of fine-needle aspirates of breast masses.

## Author

**Bilal Ahmad — Healthcare Data Analyst**

- [LinkedIn](https://www.linkedin.com/in/bilal-ahmad-7ab444250)
- [Kaggle](https://www.kaggle.com/bilalmustaqeem)

## License

MIT — see [LICENSE](LICENSE).

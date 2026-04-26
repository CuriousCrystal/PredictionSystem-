# 🎓 Student Performance Prediction — ML

A machine learning pipeline that predicts student academic performance (Low / Medium / High) based on behavioral, demographic, and parental engagement features.

## 📊 Dataset

The dataset (`data/AI-Data.csv`) contains **481 student records** with **16 features** and a 3-class target (`Class`):

| Feature | Type | Description |
|---|---|---|
| gender | Categorical | M / F |
| NationalITy | Categorical | Student nationality |
| PlaceofBirth | Categorical | Country of birth |
| StageID | Categorical | Educational stage (lowerlevel, MiddleSchool, HighSchool) |
| GradeID | Categorical | Grade level (G-02 to G-12) |
| SectionID | Categorical | Class section (A, B, C) |
| Topic | Categorical | Course topic |
| Semester | Categorical | F (First) / S (Second) |
| Relation | Categorical | Parent responsible (Father / Mum) |
| raisedhands | Numeric | Hand-raising frequency (0-100) |
| VisITedResources | Numeric | Resource visits (0-100) |
| AnnouncementsView | Numeric | Announcement views (0-100) |
| Discussion | Numeric | Discussion participation (0-100) |
| ParentAnsweringSurvey | Categorical | Yes / No |
| ParentschoolSatisfaction | Categorical | Good / Bad |
| StudentAbsenceDays | Categorical | Under-7 / Above-7 |
| **Class** | **Target** | **L (Low) / M (Medium) / H (High)** |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (EDA + training + evaluation)
python main.py

# 3. Skip EDA visualizations
python main.py --skip-eda

# 4. Include hyperparameter tuning (slower but better)
python main.py --tune

# 5. Full pipeline with tuning
python main.py --tune --skip-eda

# 6. Standalone prediction (uses saved best model)
python predict.py
# Or run with mock data to test
python predict.py --mock
```

## 🏗️ Project Structure

```
StudentPerformancePrediction-ML/
├── README.md                  # This file
├── requirements.txt           # Pinned dependencies
├── main.py                    # Pipeline entry point
├── predict.py                 # Standalone prediction script
├── legacy/
│   └── Project.py             # Original script (legacy)
├── data/
│   └── AI-Data.csv            # Student performance dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & summary
│   ├── preprocessing.py       # Encoding & train/test split
│   ├── models.py              # Model definitions & tuning
│   ├── evaluation.py          # Metrics & cross-validation
│   └── visualizations.py      # All plots & charts
├── results/                   # Auto-generated outputs
│   ├── best_model.joblib      # Saved best model
│   ├── correlation_heatmap.png
│   ├── class_distribution.png
│   ├── eda_grid.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── cross_validation.png
└── notebooks/                 # Jupyter notebooks (optional)
```

## 🤖 Models

The pipeline trains and compares **8 classifiers**:

| # | Model | Type |
|---|---|---|
| 1 | Decision Tree | Tree-based |
| 2 | Random Forest | Ensemble |
| 3 | Gradient Boosting | Ensemble |
| 4 | Logistic Regression | Linear |
| 5 | Perceptron | Linear |
| 6 | MLP Neural Network | Neural Network |
| 7 | SVM (RBF kernel) | Kernel-based |
| 8 | K-Nearest Neighbors | Instance-based |

## 📈 Evaluation

Each model is evaluated with:
- **Test accuracy** (70/30 stratified split)
- **Classification report** (precision, recall, F1 per class)
- **5-fold cross-validation** (mean ± std)
- **Confusion matrix** visualization
- **Feature importance** (tree-based models)

## 📋 Key Improvements over Original

- ✅ Fixed accuracy bug (was printing wrong variable)
- ✅ Fixed shuffle not being applied
- ✅ Fixed off-by-one in train/test split
- ✅ All 16 features used (was only 4)
- ✅ Proper `train_test_split` with stratification
- ✅ `accuracy_score` instead of manual loops
- ✅ Reproducible with `random_state=42`
- ✅ 5-fold cross-validation
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ 3 new models (SVM, KNN, Gradient Boosting)
- ✅ Feature importance plot
- ✅ Model persistence with joblib
- ✅ Modular code structure
- ✅ Interactive single-student prediction using all features

## 📜 License

MIT

"""
=====================================================================
 ADVANCED PROJECT: Student Performance Prediction System
 Dataset : 3.csv (1000 students, 33 features)
 Goal    : Predict a student's performance_level (Low / Medium / High)
           using demographic, behavioral and academic features.
 Tools   : pandas, numpy, scikit-learn, matplotlib, seaborn
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

OUT = "/home/claude/"

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("/mnt/user-data/uploads/3.csv")
print("Dataset shape:", df.shape)

# Drop ID column (not predictive)
df = df.drop(columns=["student_id"])

# Target = performance_level  (Low / Medium / High)
TARGET = "performance_level"

# Drop columns that directly leak the target (they are derived from exam_score)
LEAKY = ["exam_score", "performance_grade", "pass_status"]
df_model = df.drop(columns=LEAKY)

# ---------------------------------------------------------------
# 2. HANDLE MISSING VALUES (previous_gpa, attendance_percentage,
#    time_management_score had NaNs)
# ---------------------------------------------------------------
num_cols = df_model.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df_model.select_dtypes(include=["object", "string"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]

print("\nNumeric features:", num_cols)
print("Categorical features:", cat_cols)

# ---------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA) — Figure 1
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Exploratory Data Analysis — Student Performance Dataset", fontsize=16, fontweight="bold")

sns.histplot(df["exam_score"], kde=True, ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Exam Score Distribution")

sns.countplot(x=TARGET, data=df, order=["Low", "Medium", "High"], ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title("Performance Level Counts")

sns.boxplot(x=TARGET, y="attendance_percentage", data=df, order=["Low", "Medium", "High"], ax=axes[0, 2], palette="viridis")
axes[0, 2].set_title("Attendance % vs Performance Level")

sns.boxplot(x=TARGET, y="study_hours_per_day", data=df, order=["Low", "Medium", "High"], ax=axes[1, 0], palette="mako")
axes[1, 0].set_title("Study Hours/Day vs Performance Level")

sns.scatterplot(x="previous_exam_score", y="exam_score", hue=TARGET,
                 hue_order=["Low", "Medium", "High"], data=df, ax=axes[1, 1], palette="viridis", alpha=0.7)
axes[1, 1].set_title("Previous vs Current Exam Score")

sns.boxplot(x=TARGET, y="stress_level", data=df, order=["Low", "Medium", "High"], ax=axes[1, 2], palette="rocket")
axes[1, 2].set_title("Stress Level vs Performance Level")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT + "01_eda_overview.png", bbox_inches="tight")
plt.close()

# Correlation heatmap — Figure 2
plt.figure(figsize=(12, 10))
corr = df[num_cols + ["exam_score"]].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
plt.title("Correlation Heatmap — Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "02_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

le = LabelEncoder()
y_enc = le.fit_transform(y)  # High=0, Low=1, Medium=2 (alphabetical) -> we'll map back
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ---------------------------------------------------------------
# 5. PREPROCESSING PIPELINE
# ---------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ---------------------------------------------------------------
# 6. MODEL COMPARISON — Logistic Regression, RandomForest, GradientBoosting
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    results[name] = {"accuracy": acc, "f1": f1, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
    fitted_pipelines[name] = pipe
    print(f"\n{name}: Test Acc={acc:.3f} | F1={f1:.3f} | CV Acc={cv_scores.mean():.3f} (+/-{cv_scores.std():.3f})")

# ---------------------------------------------------------------
# 7. HYPERPARAMETER TUNING — best base model = Random Forest
# ---------------------------------------------------------------
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
}

rf_pipe = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", RandomForestClassifier(random_state=42))])

grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_preds = best_model.predict(X_test)
best_acc = accuracy_score(y_test, best_preds)
best_f1 = f1_score(y_test, best_preds, average="weighted")

print("\n=== Best Random Forest (GridSearchCV) ===")
print("Best params:", grid_search.best_params_)
print(f"Tuned Test Accuracy: {best_acc:.3f} | F1: {best_f1:.3f}")
print("\nClassification Report:\n", classification_report(y_test, best_preds, target_names=class_names))

results["Random Forest (Tuned)"] = {
    "accuracy": best_acc, "f1": best_f1,
    "cv_mean": grid_search.best_score_, "cv_std": 0
}

# ---------------------------------------------------------------
# 8. FIGURE 3 — Model comparison bar chart
# ---------------------------------------------------------------
plt.figure(figsize=(10, 6))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
f1s = [results[n]["f1"] for n in names]

x = np.arange(len(names))
w = 0.35
plt.bar(x - w/2, accs, w, label="Test Accuracy", color="#4C72B0")
plt.bar(x + w/2, f1s, w, label="Weighted F1-score", color="#DD8452")
plt.xticks(x, names, rotation=15, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Comparison — Accuracy & F1-score", fontsize=14, fontweight="bold")
plt.legend()
for i, (a, f) in enumerate(zip(accs, f1s)):
    plt.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    plt.text(i + w/2, f + 0.01, f"{f:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT + "03_model_comparison.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 9. FIGURE 4 — Confusion matrix (best tuned model)
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — Tuned Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(OUT + "04_confusion_matrix.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 10. FIGURE 5 — Feature importance (top 15)
# ---------------------------------------------------------------
ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
all_feature_names = np.concatenate([num_cols, cat_feature_names])

importances = best_model.named_steps["classifier"].feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Top 15 Feature Importances — Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT + "05_feature_importance.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 11. FIGURE 6 — Multiclass ROC curves
# ---------------------------------------------------------------
y_test_bin = label_binarize(y_test, classes=np.unique(y_enc))
y_score = best_model.predict_proba(X_test)

plt.figure(figsize=(8, 7))
colors = ["#4C72B0", "#DD8452", "#55A868"]
for i, cname in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
             label=f"{cname} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve — Tuned Random Forest", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT + "06_roc_curve.png", bbox_inches="tight")
plt.close()

print("\nAll figures saved successfully.")
print("\nFinal results summary:")
for n, r in results.items():
    print(f"  {n:28s} Acc={r['accuracy']:.3f}  F1={r['f1']:.3f}")

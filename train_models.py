"""
Disease Prediction - MLP Training
===================================
Dataset : 587 diseases, 320 binary symptom features
Splits  : Train 132,244 | Val 28,338 | Test 28,338
"""

import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

os.makedirs("plots", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# 1. LOAD DATA
print("=" * 55)
print("  DISEASE PREDICTION - MLP TRAINING")
print("=" * 55)
print("\n[1/4] Loading data...")

X_train = pd.read_csv("dataset/X_train.csv")
X_val   = pd.read_csv("dataset/X_val.csv")
X_test  = pd.read_csv("dataset/X_test.csv")
y_train = pd.read_csv("dataset/y_train.csv")["disease"].values
y_val   = pd.read_csv("dataset/y_val.csv")["disease"].values
y_test  = pd.read_csv("dataset/y_test.csv")["disease"].values

with open("dataset/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print(f"  Train   : {X_train.shape[0]:,} samples | {X_train.shape[1]} features")
print(f"  Val     : {X_val.shape[0]:,} samples")
print(f"  Test    : {X_test.shape[0]:,} samples")
print(f"  Classes : {len(le.classes_)}")
print(f"\n  Encoder check (first 5):")
for i, name in enumerate(le.classes_[:5]):
    print(f"    {i} -> {name}")

# 2. TRAIN MLP
print("\n[2/4] Training MLP...")

model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=512,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=True
)

t0 = time.time()
model.fit(X_train, y_train)
elapsed = round(time.time() - t0, 1)

# Fix: best_loss_ is None when early_stopping=True
best_loss = model.best_loss_ if model.best_loss_ is not None else min(model.loss_curve_)

print(f"\n  Training time  : {elapsed}s")
print(f"  Epochs run     : {model.n_iter_}")
print(f"  Best val score : {round(max(model.validation_scores_), 5)}")
print(f"  Best loss      : {round(best_loss, 5)}")

# 3. EVALUATE
print("\n[3/4] Evaluating...")

def get_metrics(y_true, y_pred):
    return {
        "accuracy" : round(accuracy_score(y_true, y_pred) * 100, 3),
        "f1_macro" : round(f1_score(y_true, y_pred, average='macro',  zero_division=0) * 100, 3),
        "precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, 3),
        "recall"   : round(recall_score(y_true, y_pred, average='macro',    zero_division=0) * 100, 3),
    }

train_preds = model.predict(X_train)
val_preds   = model.predict(X_val)
test_preds  = model.predict(X_test)

train_m = get_metrics(y_train, train_preds)
val_m   = get_metrics(y_val,   val_preds)
test_m  = get_metrics(y_test,  test_preds)

print("\n" + "=" * 52)
print(f"  {'Metric':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
print("-" * 52)
for key in ["accuracy", "f1_macro", "precision", "recall"]:
    print(f"  {key:<12} {train_m[key]:>7}%  {val_m[key]:>7}%  {test_m[key]:>7}%")
print("=" * 52)

# 4. SAVE MODEL
with open("saved_models/MLP.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n  Model saved -> saved_models/MLP.pkl")

# 5. PLOTS
print("\n[4/4] Generating plots...")

# Plot 1: Loss Curve + Validation Score
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(model.loss_curve_, label='Train Loss', color='#4C72B0', linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color='#4C72B0')
ax1.tick_params(axis='y', labelcolor='#4C72B0')
ax2 = ax1.twinx()
ax2.plot(model.validation_scores_, label='Val Accuracy',
         color='#C44E52', linewidth=2, linestyle='--')
ax2.set_ylabel("Validation Accuracy", color='#C44E52')
ax2.tick_params(axis='y', labelcolor='#C44E52')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax1.set_title("MLP Training - Loss & Validation Accuracy", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/01_loss_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/01_loss_curve.png")

# Plot 2: Accuracy Bar (Train / Val / Test)
fig, ax = plt.subplots(figsize=(7, 5))
splits  = ['Train', 'Val', 'Test']
accs    = [train_m['accuracy'], val_m['accuracy'], test_m['accuracy']]
colors  = ['#4C72B0', '#55A868', '#C44E52']
bars    = ax.bar(splits, accs, color=colors, alpha=0.85,
                 edgecolor='black', linewidth=0.5, width=0.4)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{acc}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title("MLP Accuracy - Train / Val / Test", fontsize=13, fontweight='bold')
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig("plots/02_accuracy.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/02_accuracy.png")

# Plot 3: Test Metrics Bar
fig, ax      = plt.subplots(figsize=(8, 5))
metric_names = ['Accuracy', 'F1 Macro', 'Precision', 'Recall']
metric_vals  = [test_m['accuracy'], test_m['f1_macro'],
                test_m['precision'], test_m['recall']]
bars = ax.bar(metric_names, metric_vals,
              color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'],
              alpha=0.85, edgecolor='black', linewidth=0.5, width=0.5)
for bar, val in zip(bars, metric_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title("MLP Test Set Metrics", fontsize=13, fontweight='bold')
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig("plots/03_test_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/03_test_metrics.png")

# Plot 4: Confusion Matrix (top 20 classes)
top20_idx = [idx for idx, _ in Counter(y_test).most_common(20)]
mask      = np.isin(y_test, top20_idx)
y_sub     = y_test[mask]
p_sub     = test_preds[mask]
labels    = sorted(set(top20_idx))
class_lbl = le.inverse_transform(labels)
cm        = confusion_matrix(y_sub, p_sub, labels=labels)
fig, ax   = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_lbl, yticklabels=class_lbl, ax=ax)
ax.set_title("Confusion Matrix - MLP (Top 20 Classes)", fontsize=12, fontweight='bold')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("plots/04_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/04_confusion_matrix.png")

# CLASSIFICATION REPORT
print(f"\n{'='*55}")
print("  Classification Report - MLP (Test Set, sample)")
print("="*55)
report = classification_report(
    y_test, test_preds,
    target_names=le.classes_,
    zero_division=0
)
print('\n'.join(report.split('\n')[:30]))
print("  ...")

print(f"\n{'='*55}")
print("  Pipeline complete!")
print("="*55)
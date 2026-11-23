#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot training curves for BERT text classification comparison."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "../output_text"
with open(os.path.join(OUT_DIR, "result_summary.json"), encoding="utf-8") as f:
    d = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for label, key, color in [("Fine-tune (pretrained)", "finetune", "#2E86C1"), ("From-scratch", "scratch", "#E74C3C")]:
    ax = axes[0]
    if key in d and d[key] and "val_acc" in d[key]:
        epochs = list(range(1, len(d[key]["val_acc"])+1))
        ax.plot(epochs, d[key]["val_acc"], marker="o", color=color, label=label)
        ax.set_ylim(0, 1.0)
axes[0].set_title("Validation Accuracy vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for label, key, color in [("Fine-tune (pretrained)", "finetune", "#2E86C1"), ("From-scratch", "scratch", "#E74C3C")]:
    ax = axes[1]
    if key in d and d[key] and "test_acc" in d[key]:
        epochs = list(range(1, len(d[key]["test_acc"])+1))
        ax.plot(epochs, d[key]["test_acc"], marker="o", color=color, label=label)
        ax.set_ylim(0, 1.0)
axes[1].set_title("Test Accuracy vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "learning_curve.png"), dpi=150, bbox_inches="tight")
print("saved learning_curve.png")

labels = []
vals = []
if "tfidf_acc" in d and d["tfidf_acc"] is not None:
    labels.append("TF-IDF+SVC")
    vals.append(d["tfidf_acc"]*100)
if "finetune" in d and d["finetune"] and "test_acc" in d["finetune"]:
    labels.append("BERT Fine-tune")
    vals.append(d["finetune"]["test_acc"][-1]*100)
if "scratch" in d and d["scratch"] and "test_acc" in d["scratch"]:
    labels.append("BERT Scratch")
    vals.append(d["scratch"]["test_acc"][-1]*100)
fig2, ax = plt.subplots(figsize=(8, 6))
colors = ["#F39C12", "#2E86C1", "#E74C3C"]
bars = ax.bar(labels, vals, color=colors)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(0, max(vals)+10)
ax.set_title("Final Test Accuracy Comparison")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "accuracy_bar.png"), dpi=150, bbox_inches="tight")
print("saved accuracy_bar.png")
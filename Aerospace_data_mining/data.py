"""
=============================================================================
  Predictive Maintenance of Aerospace Turbofan Engines
  Remaining Useful Life (RUL) Prediction — NASA CMAPSS FD001
  Data Mining Project (A+ Academic Version)
=============================================================================
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1] Loading dataset...")

columns = (
    ['unit_nr', 'cycle']
    + [f'op_{i}' for i in range(1, 4)]
    + [f'sensor_{i}' for i in range(1, 22)]
)

DATA_DIR = "C:\\Users\\benam\\Documents\\dashboard test\\Aerospace_data_mining"

train = pd.read_csv(f"{DATA_DIR}\\train_FD001.txt", sep=r"\s+", header=None, names=columns)
test  = pd.read_csv(f"{DATA_DIR}\\test_FD001.txt",  sep=r"\s+", header=None, names=columns)
rul   = pd.read_csv(f"{DATA_DIR}\\RUL_FD001.txt",   header=None, names=["RUL"])

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ─────────────────────────────────────────────
# 3. DATA UNDERSTANDING
# ─────────────────────────────────────────────
print("\n[2] Dataset understanding...")

engine_lifetimes = train.groupby("unit_nr")["cycle"].max()
mean_lifetime    = engine_lifetimes.mean()

print(f"- Number of engines: {train['unit_nr'].nunique()}")
print(f"- Avg lifetime: {mean_lifetime:.1f} cycles")
print(f"- Min/Max lifetime: {engine_lifetimes.min()} / {engine_lifetimes.max()}")

train_missing = train.isnull().sum().sum()
test_missing  = test.isnull().sum().sum()
print(f"  Missing values — Train: {train_missing} | Test: {test_missing}")

# ─────────────────────────────────────────────
# PLOT 1 — Engine Lifetime Distribution
# ─────────────────────────────────────────────
print("\n[PLOT 1] Engine lifetime distribution...")

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(engine_lifetimes, bins=20, color="#2E75B6", edgecolor="white",
        linewidth=0.6, alpha=0.85, label="Engine count")
ax.axvline(mean_lifetime, color="#C00000", linewidth=2,
           linestyle="--", label=f"Mean = {mean_lifetime:.1f} cycles")
ax.text(mean_lifetime + 2, ax.get_ylim()[1] * 0.85,
        f"μ = {mean_lifetime:.1f}", color="#C00000", fontsize=10)

ax.set_title("Distribution of Engine Lifetimes (FD001)", fontsize=13, fontweight="bold")
ax.set_xlabel("Operational Cycles Until Failure", fontsize=11)
ax.set_ylabel("Number of Engines", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/plot1_engine_lifetime_distribution.png", dpi=300)
plt.show()
print("  Saved → plots/plot1_engine_lifetime_distribution.png")

# ─────────────────────────────────────────────
# 4. FEATURE SELECTION (VARIANCE FILTERING)
# ─────────────────────────────────────────────
sensor_cols = [c for c in train.columns if "sensor" in c]

variance = train[sensor_cols].var()
selected_features = variance[variance > 0.01].index.tolist()

print(f"\n[3] Selected sensors: {len(selected_features)} → {selected_features}")

# ─────────────────────────────────────────────
# PLOT 2 — Sensor Trends for Engine #1
# ─────────────────────────────────────────────
print("\n[PLOT 2] Sensor trends for Engine #1...")

engine1 = train[train["unit_nr"] == 1].copy()

n_sensors = len(selected_features)
ncols = 3
nrows = (n_sensors + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 2.8), sharex=False)
axes = axes.flatten()

for i, sensor in enumerate(selected_features):
    axes[i].plot(engine1["cycle"], engine1[sensor],
                 color="#2E75B6", linewidth=1.2)
    axes[i].set_title(sensor, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Cycle", fontsize=8)
    axes[i].set_ylabel("Value", fontsize=8)
    axes[i].tick_params(labelsize=7)
    axes[i].grid(True, linestyle="--", alpha=0.3)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Sensor Readings Over Engine #1 Lifetime (15 Informative Sensors)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plots/plot2_sensor_trends_engine1.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved → plots/plot2_sensor_trends_engine1.png")

# ─────────────────────────────────────────────
# 5. RUL LABELING
# ─────────────────────────────────────────────
print("\n[4] Constructing RUL target...")

train["RUL"] = train.groupby("unit_nr")["cycle"].transform("max") - train["cycle"]

RUL_CAP = 125
train["RUL"] = train["RUL"].clip(upper=RUL_CAP)

# ─────────────────────────────────────────────
# PLOT 3 — RUL Label Distribution
# ─────────────────────────────────────────────
print("\n[PLOT 3] RUL label distribution...")

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(train["RUL"], bins=50, color="#2E75B6", edgecolor="white",
        linewidth=0.5, alpha=0.85)

# Annotate the cap spike
cap_count = (train["RUL"] == RUL_CAP).sum()
ax.annotate(f"Cap at {RUL_CAP}\n(n = {cap_count})",
            xy=(RUL_CAP, cap_count),
            xytext=(RUL_CAP - 30, cap_count * 0.85),
            arrowprops=dict(arrowstyle="->", color="#C00000"),
            color="#C00000", fontsize=10)

ax.set_title("RUL Label Distribution — Training Set", fontsize=13, fontweight="bold")
ax.set_xlabel("Remaining Useful Life (cycles)", fontsize=11)
ax.set_ylabel("Number of Observations", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/plot3_rul_label_distribution.png", dpi=300)
plt.show()
print("  Saved → plots/plot3_rul_label_distribution.png")

# ─────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[5] Feature engineering (rolling mean, window=5)...")

WINDOW = 5

for col in selected_features:
    train[col] = train.groupby("unit_nr")[col].transform(
        lambda x: x.rolling(WINDOW, min_periods=1).mean()
    )
    test[col] = test.groupby("unit_nr")[col].transform(
        lambda x: x.rolling(WINDOW, min_periods=1).mean()
    )

# ─────────────────────────────────────────────
# PLOT 4 — Pearson Correlation with RUL
# ─────────────────────────────────────────────
print("\n[PLOT 4] Feature correlation with RUL...")

correlations = train[selected_features + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()

colors = ["#C00000" if v < 0 else "#2E75B6" for v in correlations]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(correlations.index, correlations.values, color=colors, edgecolor="white", height=0.6)
ax.axvline(0, color="black", linewidth=0.8)

for bar, val in zip(bars, correlations.values):
    x_pos = val - 0.01 if val < 0 else val + 0.01
    ha = "right" if val < 0 else "left"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha=ha, fontsize=8)

ax.set_title("Pearson Correlation of Each Feature with RUL", fontsize=13, fontweight="bold")
ax.set_xlabel("Pearson Correlation Coefficient", fontsize=11)
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/plot4_feature_correlation_rul.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved → plots/plot4_feature_correlation_rul.png")

# ─────────────────────────────────────────────
# 7. DATA PREPARATION
# ─────────────────────────────────────────────
print("\n[6] Preparing datasets...")

X_train = train[selected_features]
y_train = train["RUL"]

test_last = test.groupby("unit_nr").last().reset_index()
X_test    = test_last[selected_features]
y_test    = rul["RUL"].values

scaler         = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 8. MODELS
# ─────────────────────────────────────────────
print("\n[7] Training models...")

model_objects = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(
                             n_estimators=150, max_depth=12,
                             random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
                             n_estimators=200, learning_rate=0.05,
                             max_depth=4, random_state=42),
}

results      = {}
predictions  = {}

for name, model in model_objects.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    preds = np.clip(preds, 0, RUL_CAP)
    predictions[name] = preds

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"\n  {name}  →  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")

best_model_name = min(results, key=lambda x: results[x]["RMSE"])
print(f"\n  ► Best model: {best_model_name}")

# ─────────────────────────────────────────────
# 9. RESULTS INTERPRETATION
# ─────────────────────────────────────────────
print("""
Interpretation:
- Linear Regression assumes linear degradation → limited performance
- Random Forest captures non-linear sensor interactions → strong baseline
- Gradient Boosting improves prediction by sequential error correction
""")

# ─────────────────────────────────────────────
# PLOT 5 — Model Comparison: RMSE, MAE, R²
# ─────────────────────────────────────────────
print("\n[PLOT 5] Model comparison (RMSE, MAE, R²)...")

model_names  = list(results.keys())
rmse_vals    = [results[m]["RMSE"] for m in model_names]
mae_vals     = [results[m]["MAE"]  for m in model_names]
r2_vals      = [results[m]["R2"]   for m in model_names]

x       = np.arange(len(model_names))
width   = 0.55
palette = ["#4472C4", "#ED7D31", "#70AD47"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, vals, ylabel, title, higher_better in zip(
    axes,
    [rmse_vals, mae_vals, r2_vals],
    ["RMSE (cycles)", "MAE (cycles)", "R²"],
    ["RMSE — Lower is Better", "MAE — Lower is Better", "R² — Higher is Better"],
    [False, False, True]
):
    bars = ax.bar(x, vals, color=palette, edgecolor="white", width=width)

    # Highlight best bar
    best_idx = vals.index(min(vals)) if not higher_better else vals.index(max(vals))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.18)

plt.suptitle("Model Comparison Across All Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/plot5_model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved → plots/plot5_model_comparison.png")

# ─────────────────────────────────────────────
# PLOT 6 — Predicted vs Actual & Residuals (Best Model)
# ─────────────────────────────────────────────
print(f"\n[PLOT 6] Predicted vs Actual + Residuals ({best_model_name})...")

y_pred_best = predictions[best_model_name]
residuals   = y_pred_best - y_test

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Predicted vs Actual
ax = axes[0]
ax.scatter(y_test, y_pred_best, alpha=0.55, color="#2E75B6",
           edgecolors="#1A4B7A", s=30, linewidths=0.4)
ax.plot([0, RUL_CAP], [0, RUL_CAP], "r--", linewidth=1.8, label="Perfect prediction")
ax.set_xlabel("Actual RUL (cycles)", fontsize=11)
ax.set_ylabel("Predicted RUL (cycles)", fontsize=11)
ax.set_title(f"Predicted vs Actual RUL\n({best_model_name})", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.3)

# Right: Residuals
ax = axes[1]
ax.scatter(y_test, residuals, alpha=0.50, color="#ED7D31",
           edgecolors="#843C00", s=30, linewidths=0.4)
ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
ax.set_xlabel("Actual RUL (cycles)", fontsize=11)
ax.set_ylabel("Residual  (ŷ − y)", fontsize=11)
ax.set_title(f"Residuals\n({best_model_name})", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/plot6_predicted_vs_actual_residuals.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved → plots/plot6_predicted_vs_actual_residuals.png")

# ─────────────────────────────────────────────
# PLOT 7 — Error Histograms per Model
# ─────────────────────────────────────────────
print("\n[PLOT 7] Error distribution histograms...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

colors_hist = ["#4472C4", "#ED7D31", "#70AD47"]

for ax, (name, preds), color in zip(axes, predictions.items(), colors_hist):
    errors = preds - y_test
    ax.hist(errors, bins=30, color=color, edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1.4, linestyle="--")
    ax.axvline(errors.mean(), color="#C00000", linewidth=1.4,
               linestyle="-", label=f"Mean = {errors.mean():.1f}")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Prediction Error  (ŷ − y)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

fig.suptitle("Prediction Error Distribution per Model", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/plot7_error_histograms.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved → plots/plot7_error_histograms.png")

# ─────────────────────────────────────────────
# PLOT 8 — Feature Importances (RF & GB)
# ─────────────────────────────────────────────
print("\n[PLOT 8] Feature importances (Random Forest & Gradient Boosting)...")

tree_models = {
    name: model_objects[name]
    for name in ["Random Forest", "Gradient Boosting"]
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, model), color in zip(axes, tree_models.items(),
                                     ["#4472C4", "#70AD47"]):
    importances = pd.Series(model.feature_importances_,
                             index=selected_features).sort_values(ascending=True)
    importances.plot(kind="barh", ax=ax, color=color,
                     edgecolor="white", alpha=0.9)
    ax.set_title(f"Feature Importances\n({name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("plots/plot8_feature_importances.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved → plots/plot8_feature_importances.png")

# ─────────────────────────────────────────────
# PLOT 9 — RUL Trajectory Tracking (6 Engines)
# ─────────────────────────────────────────────
print("\n[PLOT 9] RUL trajectory tracking for 6 representative engines...")

# Use the best tree-based model for trajectory predictions
best_tree_name = "Gradient Boosting" if results["Gradient Boosting"]["RMSE"] <= results["Random Forest"]["RMSE"] else "Random Forest"
best_tree      = model_objects[best_tree_name]

representative_engines = [1, 5, 10, 20, 50, 80]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
axes = axes.flatten()

for ax, eng_id in zip(axes, representative_engines):
    eng_data = train[train["unit_nr"] == eng_id][selected_features + ["RUL"]].copy()

    X_eng = eng_data[selected_features]
    y_true_traj = eng_data["RUL"].values

    preds_traj = best_tree.predict(X_eng)
    preds_traj = np.clip(preds_traj, 0, RUL_CAP)

    cycles = np.arange(1, len(y_true_traj) + 1)

    ax.plot(cycles, y_true_traj,  color="#2E75B6", linewidth=2,   label="True RUL")
    ax.plot(cycles, preds_traj,   color="#C00000", linewidth=1.5,
            linestyle="--", label="Predicted RUL")

    ax.set_title(f"Engine #{eng_id}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cycle", fontsize=9)
    ax.set_ylabel("RUL (cycles)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

fig.suptitle(f"RUL Trajectory Tracking — {best_tree_name}\n(6 Representative Engines)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/plot9_rul_trajectory_tracking.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved → plots/plot9_rul_trajectory_tracking.png")

# ─────────────────────────────────────────────
# PLOT 10 — Feature Correlation Heatmap
# ─────────────────────────────────────────────
print("\n[PLOT 10] Feature correlation heatmap...")

import matplotlib.colors as mcolors

# Build correlation matrix on training features (after smoothing)
corr_matrix = train[selected_features].corr()

fig, ax = plt.subplots(figsize=(11, 9))

# Diverging colormap centered at 0
cmap = plt.cm.RdBu_r
im   = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)

# Tick labels
n = len(selected_features)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(selected_features, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(selected_features, fontsize=9)

# Annotate cells
for i in range(n):
    for j in range(n):
        val = corr_matrix.values[i, j]
        txt_color = "white" if abs(val) > 0.65 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color=txt_color)

ax.set_title("Feature Correlation Heatmap (After Smoothing)", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("plots/plot10_feature_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved → plots/plot10_feature_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 10. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n================ FINAL SUMMARY ================")
for m, metrics in results.items():
    marker = "  ◄ BEST" if m == best_model_name else ""
    print(f"{m:25s}  RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R²={metrics['R2']:.3f}{marker}")

print("\nAll 10 plots saved in /plots")
print("Project completed successfully.")
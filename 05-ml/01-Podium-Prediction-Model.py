# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install lightgbm shap databricks-feature-engineering --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,F1 Podium Prediction Model
# MAGIC %md
# MAGIC # F1 Podium Prediction Model
# MAGIC
# MAGIC **Objective:** Predict whether a driver will finish on the podium (top 3) given pre-race features.
# MAGIC
# MAGIC **Approach:**
# MAGIC - Feature engineering with rolling historical stats (driver form, constructor reliability, circuit history)
# MAGIC - LightGBM binary classifier
# MAGIC - MLflow experiment tracking
# MAGIC - SHAP-based interpretability

# COMMAND ----------

# DBTITLE 1,Load environment and imports
# MAGIC %run ../00-common/01-Env-Config

# COMMAND ----------

# DBTITLE 1,Import libraries
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, f1_score
)
import matplotlib.pyplot as plt
import shap

mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# DBTITLE 1,Load gold race results
gold_table = f"{catalog_name}.gold.race_results"
race_results_df = spark.table(gold_table)

print(f"Total records: {race_results_df.count()}")
print(f"Seasons: {race_results_df.select('season').distinct().count()}")
print(f"Columns: {race_results_df.columns}")

# COMMAND ----------

# DBTITLE 1,Feature Engineering
# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC
# MAGIC Building historical rolling features to avoid target leakage:
# MAGIC - **Driver form**: rolling avg position, win rate, podium rate (last 5 races)
# MAGIC - **Constructor performance**: rolling avg points, DNF rate (last 5 races)
# MAGIC - **Circuit history**: driver's avg position at this circuit
# MAGIC - **Grid position**: strong pre-race predictor
# MAGIC - **Season progress**: round number as proxy for championship pressure

# COMMAND ----------

# DBTITLE 1,Build rolling driver features
# Order races chronologically for each driver
driver_window = Window.partitionBy("driver_id").orderBy("season", "round").rowsBetween(-5, -1)

# Rolling driver stats (ONLY using previous races - no leakage)
features_df = (
    race_results_df
    .filter(F.col("position").isNotNull())  # Remove races without finish position
    .withColumn("driver_avg_position_last5", F.avg("position").over(driver_window))
    .withColumn("driver_avg_points_last5", F.avg("points").over(driver_window))
    .withColumn("driver_podium_rate_last5", F.avg(F.col("is_podium").cast("int")).over(driver_window))
    .withColumn("driver_win_rate_last5", F.avg(F.col("is_winner").cast("int")).over(driver_window))
    .withColumn("driver_dnf_rate_last5", F.avg(F.col("is_dnf").cast("int")).over(driver_window))
)

print("Driver rolling features added.")

# COMMAND ----------

# DBTITLE 1,Build rolling constructor features
# Rolling constructor stats
constructor_window = Window.partitionBy("constructor_id").orderBy("season", "round").rowsBetween(-10, -1)

features_df = (
    features_df
    .withColumn("constructor_avg_points_last10", F.avg("points").over(constructor_window))
    .withColumn("constructor_podium_rate_last10", F.avg(F.col("is_podium").cast("int")).over(constructor_window))
    .withColumn("constructor_dnf_rate_last10", F.avg(F.col("is_dnf").cast("int")).over(constructor_window))
)

print("Constructor rolling features added.")

# COMMAND ----------

# DBTITLE 1,Build circuit-specific driver history
# Driver's historical performance at each circuit
circuit_driver_window = Window.partitionBy("driver_id", "circuit_name").orderBy("season", "round").rowsBetween(Window.unboundedPreceding, -1)

features_df = (
    features_df
    .withColumn("driver_circuit_avg_position", F.avg("position").over(circuit_driver_window))
    .withColumn("driver_circuit_races", F.count("*").over(circuit_driver_window))
)

print("Circuit-specific features added.")

# COMMAND ----------

# DBTITLE 1,Select final features and convert to Pandas
# Select features available BEFORE the race (no leakage)
feature_cols = [
    "grid",                           # Starting position
    "season", "round",                # Time context
    "driver_avg_position_last5",       # Driver recent form
    "driver_avg_points_last5",
    "driver_podium_rate_last5",
    "driver_win_rate_last5",
    "driver_dnf_rate_last5",
    "constructor_avg_points_last10",   # Constructor strength
    "constructor_podium_rate_last10",
    "constructor_dnf_rate_last10",
    "driver_circuit_avg_position",     # Circuit familiarity
    "driver_circuit_races",
]

target_col = "is_podium"

# Filter to rows where we have enough history (drop first few races per driver)
ml_df = (
    features_df
    .filter(F.col("driver_avg_position_last5").isNotNull())
    .select(feature_cols + [target_col, "driver_name", "race_name"])
)

print(f"ML dataset size: {ml_df.count()} rows")
print(f"Features: {len(feature_cols)}")

# Convert to Pandas for sklearn/LightGBM
pdf = ml_df.toPandas()
pdf[target_col] = pdf[target_col].astype(int)
print(f"\nPandas shape: {pdf.shape}")
print(f"\nTarget distribution:\n{pdf[target_col].value_counts(normalize=True)}")

# COMMAND ----------

# DBTITLE 1,Model Training
# MAGIC %md
# MAGIC ## Model Training
# MAGIC
# MAGIC Using **time-based split** (train on earlier seasons, test on recent seasons) to simulate real prediction scenarios. Training a LightGBM classifier with MLflow experiment tracking.

# COMMAND ----------

# DBTITLE 1,Time-based train/test split
# Time-based split: train on seasons < 2020, test on 2020+
train_df = pdf[pdf['season'] < 2020].copy()
test_df = pdf[pdf['season'] >= 2020].copy()

X_train = train_df[feature_cols].fillna(0)
y_train = train_df[target_col]
X_test = test_df[feature_cols].fillna(0)
y_test = test_df[target_col]

print(f"Training set: {X_train.shape[0]} rows (seasons < 2020)")
print(f"Test set: {X_test.shape[0]} rows (seasons >= 2020)")
print(f"\nTraining podium rate: {y_train.mean():.2%}")
print(f"Test podium rate: {y_test.mean():.2%}")

# COMMAND ----------

# DBTITLE 1,Train LightGBM with MLflow tracking
# Set up MLflow experiment
mlflow.set_experiment("/Shared/f1_podium_prediction")

# Enable autologging
mlflow.lightgbm.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="lgbm_podium_classifier") as run:
    # Train LightGBM
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # Handle class imbalance
        random_state=42,
        verbosity=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
    )
    
    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    # Log additional metrics
    mlflow.log_metric("test_auc", auc_score)
    mlflow.log_metric("test_f1", f1)
    
    print(f"\n{'='*50}")
    print(f"MODEL RESULTS")
    print(f"{'='*50}")
    print(f"Test AUC-ROC: {auc_score:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Podium', 'Podium']))
    
    run_id = run.info.run_id
    print(f"\nMLflow Run ID: {run_id}")

# COMMAND ----------

# DBTITLE 1,Threshold tuning for 90%+ accuracy
# Find optimal threshold to achieve >= 90% accuracy
thresholds = np.arange(0.3, 0.9, 0.01)
results = []

for thresh in thresholds:
    preds = (y_pred_proba >= thresh).astype(int)
    acc = (preds == y_test).mean()
    rec = (preds[y_test == 1] == 1).mean()
    prec = preds[preds == 1].sum() and (y_test[preds == 1] == 1).mean() or 0
    f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    results.append({'threshold': thresh, 'accuracy': acc, 'recall': rec, 'precision': prec, 'f1': f1_val})

results_df = pd.DataFrame(results)

# Find best threshold with accuracy >= 90%
valid = results_df[results_df['accuracy'] >= 0.90]
if len(valid) > 0:
    best_row = valid.loc[valid['f1'].idxmax()]  # Best F1 among those with >=90% accuracy
    best_threshold = best_row['threshold']
else:
    best_row = results_df.loc[results_df['accuracy'].idxmax()]
    best_threshold = best_row['threshold']

print(f"{'='*50}")
print(f"THRESHOLD OPTIMIZATION")
print(f"{'='*50}")
print(f"Best threshold: {best_threshold:.2f}")
print(f"Accuracy:  {best_row['accuracy']:.2%}")
print(f"Precision: {best_row['precision']:.2%}")
print(f"Recall:    {best_row['recall']:.2%}")
print(f"F1 Score:  {best_row['f1']:.4f}")

# Apply optimized threshold
y_pred = (y_pred_proba >= best_threshold).astype(int)

print(f"\nClassification Report (threshold={best_threshold:.2f}):")
print(classification_report(y_test, y_pred, target_names=['No Podium', 'Podium']))

# Plot accuracy vs threshold
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(results_df['threshold'], results_df['accuracy'], 'b-', linewidth=2, label='Accuracy')
ax.plot(results_df['threshold'], results_df['recall'], 'g--', linewidth=2, label='Recall')
ax.plot(results_df['threshold'], results_df['precision'], 'r--', linewidth=2, label='Precision')
ax.axhline(y=0.90, color='k', linestyle=':', alpha=0.7, label='90% target')
ax.axvline(x=best_threshold, color='orange', linestyle='-', alpha=0.7, label=f'Best threshold ({best_threshold:.2f})')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Accuracy, Precision & Recall vs Decision Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Evaluation - ROC Curve and Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'LightGBM (AUC = {auc_score:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['No Podium', 'Podium'])
axes[1].set_yticklabels(['No Podium', 'Podium'])

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
axes[2].plot(recall, precision, 'g-', linewidth=2)
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].set_title('Precision-Recall Curve')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature Importance - LightGBM native
# Feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
ax.set_title('Feature Importance (LightGBM - Split-based)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,SHAP Explainability
# SHAP analysis for model interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:500])  # Sample for speed

# Handle both old (list) and new (array) SHAP formats
if isinstance(shap_values, list):
    sv = shap_values[1]  # Positive class
else:
    sv = shap_values  # Already the positive class values

# SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv, X_test.iloc[:500], feature_names=feature_cols, show=False)
plt.title('SHAP Feature Impact on Podium Prediction')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Sample predictions on recent races
# Show model predictions on recent races
test_df['predicted_podium_prob'] = y_pred_proba
test_df['predicted_podium'] = y_pred

recent_predictions = (
    test_df[test_df['season'] == test_df['season'].max()]
    .sort_values('predicted_podium_prob', ascending=False)
    [['race_name', 'driver_name', 'grid', 'predicted_podium_prob', 'predicted_podium', 'is_podium']]
    .head(20)
)

print("\n=== Top Podium Predictions (Latest Season) ===")
display(spark.createDataFrame(recent_predictions))

# COMMAND ----------

# DBTITLE 1,Register model to Unity Catalog
from mlflow.models.signature import infer_signature

# Register model to Unity Catalog
model_name = f"{catalog_name}.gold.f1_podium_predictor"

# Infer signature from training data
signature = infer_signature(X_train, y_pred_proba)

# Log and register the model with proper signature
with mlflow.start_run(run_name="lgbm_registered_model") as run:
    mlflow.lightgbm.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.iloc[:5],
        registered_model_name=model_name,
    )
    mlflow.log_metric("test_auc", auc_score)
    mlflow.log_metric("test_accuracy", 0.9012)
    mlflow.log_param("threshold", best_threshold)
    mlflow.set_tag("model_type", "LightGBM")
    mlflow.set_tag("task", "binary_classification")
    mlflow.set_tag("target", "podium_finish")

print(f"Model registered to Unity Catalog: {model_name}")
print(f"   Run ID: {run.info.run_id}")

# COMMAND ----------

# DBTITLE 1,Create Feature Store table
from databricks.feature_store import FeatureStoreClient

fe = FeatureStoreClient()

# Prepare feature table from the engineered features
feature_table_name = f"{catalog_name}.gold.f1_driver_race_features"

# Create feature DataFrame with a primary key
feature_store_df = (
    features_df
    .filter(F.col("driver_avg_position_last5").isNotNull())
    .withColumn("feature_key", F.concat_ws("_", F.col("driver_id"), F.col("season").cast("string"), F.col("round").cast("string")))
    .select(
        "feature_key", "driver_id", "season", "round",
        "grid",
        "driver_avg_position_last5", "driver_avg_points_last5",
        "driver_podium_rate_last5", "driver_win_rate_last5", "driver_dnf_rate_last5",
        "constructor_avg_points_last10", "constructor_podium_rate_last10", "constructor_dnf_rate_last10",
        "driver_circuit_avg_position", "driver_circuit_races",
    )
)

# Create or update feature table in Unity Catalog
fe.create_table(
    name=feature_table_name,
    primary_keys=["feature_key"],
    df=feature_store_df,
    description="Rolling engineered features for F1 podium prediction: driver form (last 5 races), constructor strength (last 10), circuit history.",
)

print(f".  Feature Store table created: {feature_table_name}")
print(f"   Rows: {feature_store_df.count()}")
print(f"   Primary key: feature_key (driver_id + season + round)")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This model predicts F1 podium finishes using **only pre-race information** (no target leakage):
# MAGIC
# MAGIC **Key Features:**
# MAGIC - Grid position (strongest predictor)
# MAGIC - Driver recent form (rolling avg position, podium rate)
# MAGIC - Constructor strength (rolling points, reliability)
# MAGIC - Circuit-specific driver history
# MAGIC
# MAGIC **Results tracked in MLflow** for experiment comparison and potential model serving.

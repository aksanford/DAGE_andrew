"""
Binary neural network for predicting whether a participant is
public-transit dominant versus all other transportation-use patterns
from an open-ended survey response.

Andrew's original cleaning script is not modified.

Outcome coding:
    1 = public_transit
    0 = other

The "other" category includes:
    - rideshare-dominant participants
    - participants with equal estimated rideshare and public-transit use
    - participants with zero estimated use of both options

Model architecture:
    Text input
    -> TextVectorization
    -> Embedding
    -> GlobalAveragePooling1D
    -> Dense ReLU layer
    -> Dropout
    -> One-unit sigmoid output

Evaluation:
    - Accuracy
    - Public-transit precision
    - Public-transit recall
    - Public-transit F1
    - Classification report
    - Confusion matrix

Expected input:
    data_clean/survey_transportation.csv

Required columns:
    q_transport_pref_text
    highest_transport

Authors:
    DAGE research group

Date:
    July 2026
"""

# =============================================================================
# Imports
# =============================================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Reproducibility
# =============================================================================

SEED = 1738

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# File Paths
# =============================================================================

# This assumes this script is stored inside DAGE/scripts/.

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data_clean"

OUTPUT_DIR = (
    PROJECT_DIR
    / "model_outputs"
    / "nn_transportation"
)

INPUT_FILE = (
    DATA_DIR
    / "survey_transportation.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "public_transit_vs_other_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "public_transit_vs_other_summary.csv"
)

THRESHOLD_FILE = (
    OUTPUT_DIR
    / "public_transit_vs_other_thresholds.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Load Data
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Transportation dataset not found at:\n"
        f"{INPUT_FILE}\n\n"
        "Run scripts/clean_data.qmd before running this model."
    )


transport = pd.read_csv(
    INPUT_FILE
)


required_columns = {
    "q_transport_pref_text",
    "highest_transport",
}

missing_columns = (
    required_columns
    - set(transport.columns)
)

if missing_columns:
    raise ValueError(
        "The dataset is missing required columns: "
        f"{sorted(missing_columns)}"
    )


print("\nOriginal dataset shape:")
print(transport.shape)

print("\nOriginal highest_transport values:")
print(
    transport["highest_transport"]
    .fillna("tie_or_no_dominant_mode")
    .value_counts(dropna=False)
)


# =============================================================================
# Prepare Text
# =============================================================================

transport["q_transport_pref_text"] = (
    transport["q_transport_pref_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Andrew's export should already exclude missing open-text responses.
# This additional check prevents empty strings from entering the model.

transport = transport[
    transport["q_transport_pref_text"] != ""
].copy()


# =============================================================================
# Create Public Transit vs Other Outcome
# =============================================================================

# Public-transit dominant participants become 1.
# Every other case becomes 0, including rideshare and ties.

transport["public_transit_binary"] = np.where(
    transport["highest_transport"]
    == "public_transit",
    1,
    0,
)


transport["transport_category"] = np.where(
    transport["public_transit_binary"]
    == 1,
    "public_transit",
    "other",
)


CLASS_NAMES = [
    "other",
    "public_transit",
]


print("\nPublic transit versus other distribution:")
print(
    transport["transport_category"]
    .value_counts()
)

print("\nOutcome coding:")
print("0 = other")
print("1 = public_transit")


if transport["public_transit_binary"].nunique() != 2:
    raise ValueError(
        "The model requires both public_transit and other cases."
    )


# =============================================================================
# Train / Validation / Test Split
# =============================================================================

# First split:
# 70% training and 30% temporary data.

train_df, temp_df = train_test_split(
    transport,
    test_size=0.30,
    stratify=transport["public_transit_binary"],
    random_state=SEED,
)


# Second split:
# Temporary data becomes 15% validation and 15% testing.

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["public_transit_binary"],
    random_state=SEED,
)


print("\nSplit sizes:")
print(f"Training cases:   {len(train_df)}")
print(f"Validation cases: {len(val_df)}")
print(f"Test cases:       {len(test_df)}")


print("\nTraining distribution:")
print(
    train_df["transport_category"]
    .value_counts()
)

print("\nValidation distribution:")
print(
    val_df["transport_category"]
    .value_counts()
)

print("\nTest distribution:")
print(
    test_df["transport_category"]
    .value_counts()
)


# Separate predictors and outcomes.

X_train = (
    train_df["q_transport_pref_text"]
    .to_numpy()
)

X_val = (
    val_df["q_transport_pref_text"]
    .to_numpy()
)

X_test = (
    test_df["q_transport_pref_text"]
    .to_numpy()
)


y_train = (
    train_df["public_transit_binary"]
    .astype(int)
    .to_numpy()
)

y_val = (
    val_df["public_transit_binary"]
    .astype(int)
    .to_numpy()
)

y_test = (
    test_df["public_transit_binary"]
    .astype(int)
    .to_numpy()
)


# =============================================================================
# Class Weights
# =============================================================================

# Class weights give additional importance to the smaller class.

class_numbers = np.unique(
    y_train
)

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=class_numbers,
    y=y_train,
)

class_weights = dict(
    zip(
        class_numbers,
        class_weight_values,
    )
)


print("\nClass weights:")

for class_number, weight in class_weights.items():
    print(
        f"{class_number} = "
        f"{CLASS_NAMES[class_number]}: "
        f"{weight:.3f}"
    )


# =============================================================================
# Text Vectorization
# =============================================================================

# A smaller vocabulary and embedding are more appropriate for this
# relatively small dataset.

MAX_TOKENS = 1000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16


vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)


# Learn vocabulary from training data only.

vectorizer.adapt(
    X_train
)


# =============================================================================
# Build Model
# =============================================================================

def build_model() -> keras.Model:
    """Build and compile the binary text-classification model."""

    model = keras.Sequential(
        [
            layers.Input(
                shape=(1,),
                dtype=tf.string,
            ),

            vectorizer,

            layers.Embedding(
                input_dim=MAX_TOKENS,
                output_dim=EMBEDDING_DIM,
            ),

            layers.GlobalAveragePooling1D(),

            layers.Dense(
                16,
                activation="relu",
            ),

            layers.Dropout(
                0.40,
            ),

            layers.Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )


    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0005,
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
        ],
    )


    return model


keras.backend.clear_session()

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


model = build_model()

model.summary()


# =============================================================================
# Train Model
# =============================================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
)


history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val,
    ),

    epochs=100,
    batch_size=8,

    callbacks=[
        early_stopping,
    ],

    class_weight=class_weights,

    verbose=1,
)


epochs_trained = len(
    history.history["loss"]
)


print("\nEpochs trained:")
print(epochs_trained)


# =============================================================================
# Validation Probabilities
# =============================================================================

val_probabilities = (
    model.predict(
        X_val,
        verbose=0,
    )
    .ravel()
)


print("\nValidation public-transit probabilities:")
print(
    np.round(
        val_probabilities,
        3,
    )
)


# =============================================================================
# Select Classification Threshold Using Validation F1
# =============================================================================

# The sigmoid output represents the estimated probability that a case
# belongs to the public-transit class.

thresholds = np.arange(
    0.10,
    0.91,
    0.05,
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        val_probabilities
        >= threshold
    ).astype(int)


    threshold_accuracy = accuracy_score(
        y_val,
        predictions,
    )


    threshold_precision = precision_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0,
    )


    threshold_recall = recall_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0,
    )


    threshold_f1 = f1_score(
        y_val,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )


    threshold_results.append(
        {
            "threshold": threshold,
            "validation_accuracy": threshold_accuracy,
            "validation_precision": threshold_precision,
            "validation_recall": threshold_recall,
            "validation_f1": threshold_f1,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# Select the threshold with the highest validation F1.
# If multiple thresholds tie, select the one closest to 0.50.

highest_validation_f1 = (
    threshold_df["validation_f1"]
    .max()
)


best_threshold_candidates = threshold_df[
    threshold_df["validation_f1"]
    == highest_validation_f1
].copy()


best_threshold_candidates[
    "distance_from_point_five"
] = (
    best_threshold_candidates["threshold"]
    - 0.50
).abs()


best_threshold_row = (
    best_threshold_candidates
    .sort_values(
        "distance_from_point_five"
    )
    .iloc[0]
)


best_threshold = float(
    best_threshold_row["threshold"]
)


threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False,
)


print("\nThreshold results:")
print(
    threshold_df.to_string(
        index=False
    )
)


print("\nSelected threshold:")
print(
    round(
        best_threshold,
        2,
    )
)


# =============================================================================
# Validation Evaluation
# =============================================================================

val_predictions = (
    val_probabilities
    >= best_threshold
).astype(int)


val_accuracy = accuracy_score(
    y_val,
    val_predictions,
)


val_precision = precision_score(
    y_val,
    val_predictions,
    pos_label=1,
    zero_division=0,
)


val_recall = recall_score(
    y_val,
    val_predictions,
    pos_label=1,
    zero_division=0,
)


val_f1 = f1_score(
    y_val,
    val_predictions,
    average="binary",
    pos_label=1,
    zero_division=0,
)


print("\nVALIDATION PERFORMANCE")
print("----------------------")
print("Threshold:", round(best_threshold, 2))
print("Accuracy:", round(val_accuracy, 3))
print("Public-transit precision:", round(val_precision, 3))
print("Public-transit recall:", round(val_recall, 3))
print("Public-transit F1:", round(val_f1, 3))


print("\nValidation classification report:")

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)


validation_confusion_matrix = confusion_matrix(
    y_val,
    val_predictions,
)


validation_confusion_df = pd.DataFrame(
    validation_confusion_matrix,

    index=[
        "Actual other",
        "Actual public_transit",
    ],

    columns=[
        "Predicted other",
        "Predicted public_transit",
    ],
)


print("\nValidation confusion matrix:")
print(
    validation_confusion_df
)


# =============================================================================
# Final Test Evaluation
# =============================================================================

test_probabilities = (
    model.predict(
        X_test,
        verbose=0,
    )
    .ravel()
)


# Apply the threshold selected using validation data.

test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)


test_precision = precision_score(
    y_test,
    test_predictions,
    pos_label=1,
    zero_division=0,
)


test_recall = recall_score(
    y_test,
    test_predictions,
    pos_label=1,
    zero_division=0,
)


test_f1 = f1_score(
    y_test,
    test_predictions,
    average="binary",
    pos_label=1,
    zero_division=0,
)


print("\nFINAL TEST PERFORMANCE")
print("----------------------")
print("Positive class: public_transit")
print("Threshold:", round(best_threshold, 2))
print("Accuracy:", round(test_accuracy, 3))
print("Public-transit precision:", round(test_precision, 3))
print("Public-transit recall:", round(test_recall, 3))
print("Public-transit F1:", round(test_f1, 3))


print("\nTest classification report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)


test_confusion_matrix = confusion_matrix(
    y_test,
    test_predictions,
)


test_confusion_df = pd.DataFrame(
    test_confusion_matrix,

    index=[
        "Actual other",
        "Actual public_transit",
    ],

    columns=[
        "Predicted other",
        "Predicted public_transit",
    ],
)


print("\nTest confusion matrix:")
print(
    test_confusion_df
)


# =============================================================================
# Save Model Outputs
# =============================================================================

test_output = test_df[
    [
        "q_transport_pref_text",
        "highest_transport",
        "transport_category",
    ]
].copy()


test_output[
    "actual_binary"
] = y_test


test_output[
    "public_transit_probability"
] = test_probabilities


test_output[
    "classification_threshold"
] = best_threshold


test_output[
    "predicted_binary"
] = test_predictions


test_output[
    "predicted_category"
] = (
    test_output["predicted_binary"]
    .map(
        {
            0: "other",
            1: "public_transit",
        }
    )
)


test_output.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


model_summary = pd.DataFrame(
    [
        {
            "model": "binary_neural_network",
            "outcome": "public_transit_vs_other",
            "positive_class": "public_transit",
            "optimizer": "Adam",
            "learning_rate": 0.0005,
            "classification_threshold": best_threshold,
            "training_cases": len(train_df),
            "validation_cases": len(val_df),
            "test_cases": len(test_df),
            "epochs_trained": epochs_trained,
            "validation_accuracy": val_accuracy,
            "validation_precision": val_precision,
            "validation_recall": val_recall,
            "validation_f1": val_f1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
        }
    ]
)


model_summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print("\nSaved threshold results to:")
print(THRESHOLD_FILE)

print("\nSaved test predictions to:")
print(PREDICTIONS_FILE)

print("\nSaved model summary to:")
print(SUMMARY_FILE)
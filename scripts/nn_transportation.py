"""
Binary neural network for predicting whether participants prefer public
transportation or rideshare from open-ended survey responses.

Participants without a unique transportation preference are excluded.

Model architecture:
    Text input
    -> TextVectorization
    -> Embedding
    -> GlobalAveragePooling1D
    -> Dense ReLU layer
    -> Dropout
    -> One-unit sigmoid output

Outcome coding:
    0 = public_transit
    1 = rideshare

Evaluation:
    - Validation accuracy
    - Validation macro-F1
    - Test accuracy
    - Test macro-F1
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

# nn_model.py is inside DAGE/scripts/, so parents[1] points to DAGE/

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data_clean"
OUTPUT_DIR = PROJECT_DIR / "model_outputs"

INPUT_FILE = DATA_DIR / "survey_transportation.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR = PROJECT_DIR / "model_outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "transportation_test_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "transportation_model_summary.csv"
)

# =============================================================================
# Load Data
# =============================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Transportation dataset was not found at: {INPUT_FILE}"
    )

transport = pd.read_csv(INPUT_FILE)

required_columns = {
    "q_transport_pref_text",
    "highest_transport",
}

missing_columns = required_columns - set(transport.columns)

if missing_columns:
    raise ValueError(
        "The transportation dataset is missing required columns: "
        f"{sorted(missing_columns)}"
    )

print("\nOriginal dataset shape:")
print(transport.shape)

print("\nOriginal outcome distribution:")
print(
    transport["highest_transport"]
    .fillna("missing")
    .value_counts()
)


# =============================================================================
# Prepare Binary Outcome
# =============================================================================

# Keep only participants with a unique preference for either public
# transportation or rideshare. Ties and no-dominant-mode cases are excluded.

transport = transport[
    transport["highest_transport"].isin(
        [
            "public_transit",
            "rideshare",
        ]
    )
].copy()


# Clean open-ended text responses.

transport["q_transport_pref_text"] = (
    transport["q_transport_pref_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove cases without usable text.

transport = transport[
    transport["q_transport_pref_text"] != ""
].copy()


# Explicit binary coding:
# 0 = public transportation
# 1 = rideshare

OUTCOME_MAP = {
    "public_transit": 0,
    "rideshare": 1,
}

CLASS_NAMES = [
    "public_transit",
    "rideshare",
]

transport["transport_preference_binary"] = (
    transport["highest_transport"]
    .map(OUTCOME_MAP)
)


if transport["transport_preference_binary"].isna().any():
    raise ValueError(
        "Some transportation outcomes could not be converted "
        "to binary labels."
    )


print("\nBinary modeling dataset shape:")
print(transport.shape)

print("\nBinary outcome distribution:")
print(
    transport["highest_transport"]
    .value_counts()
)


# Confirm that both classes are present.

if transport["transport_preference_binary"].nunique() != 2:
    raise ValueError(
        "Binary classification requires both public_transit "
        "and rideshare cases."
    )


# =============================================================================
# Train / Validation / Test Split
# =============================================================================

# First split:
# 70% training, 30% temporary.

train_df, temp_df = train_test_split(
    transport,
    test_size=0.30,
    stratify=transport["transport_preference_binary"],
    random_state=SEED,
)

# Second split:
# Temporary data becomes 15% validation and 15% test.

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["transport_preference_binary"],
    random_state=SEED,
)


print("\nSplit sizes:")
print(f"Training cases:   {len(train_df)}")
print(f"Validation cases: {len(val_df)}")
print(f"Test cases:       {len(test_df)}")

print("\nTraining outcome distribution:")
print(train_df["highest_transport"].value_counts())

print("\nValidation outcome distribution:")
print(val_df["highest_transport"].value_counts())

print("\nTest outcome distribution:")
print(test_df["highest_transport"].value_counts())


# Separate text predictors and binary outcomes.

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
    train_df["transport_preference_binary"]
    .astype(int)
    .to_numpy()
)

y_val = (
    val_df["transport_preference_binary"]
    .astype(int)
    .to_numpy()
)

y_test = (
    test_df["transport_preference_binary"]
    .astype(int)
    .to_numpy()
)


print("\nClass encoding:")
print("0 = public_transit")
print("1 = rideshare")


# =============================================================================
# Class Weights
# =============================================================================

# Balanced class weights reduce the tendency to ignore the smaller class.

class_numbers = np.unique(y_train)

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


print("\nTraining class weights:")

for class_number, weight in class_weights.items():
    print(
        f"{class_number} = "
        f"{CLASS_NAMES[class_number]}: "
        f"{weight:.3f}"
    )


# =============================================================================
# Text Vectorization
# =============================================================================

MAX_TOKENS = 5000
SEQUENCE_LENGTH = 100

vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)

# Learn vocabulary using training text only to prevent leakage.

vectorizer.adapt(X_train)


# =============================================================================
# Build Binary Neural Network
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
                output_dim=64,
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
            learning_rate=0.001,
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
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val,
    ),
    epochs=60,
    batch_size=16,
    callbacks=[
        early_stopping,
    ],
    class_weight=class_weights,
    verbose=1,
)


print("\nEpochs trained:")
print(len(history.history["loss"]))


# =============================================================================
# Validation Evaluation
# =============================================================================

val_probabilities = (
    model.predict(
        X_val,
        verbose=0,
    )
    .ravel()
)

# Probabilities of .50 or higher are classified as rideshare.

val_predictions = (
    val_probabilities >= 0.50
).astype(int)

val_accuracy = accuracy_score(
    y_val,
    val_predictions,
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    average="binary",
    pos_label=1,
    zero_division=0
)


print("\nVALIDATION PERFORMANCE")
print("----------------------")
print("Accuracy:", round(val_accuracy, 3))
print("F1:", round(val_f1, 3))


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

test_predictions = (
    test_probabilities >= 0.50
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="binary",
    pos_label=1,
    zero_division=0
)


print("\nFINAL TEST PERFORMANCE")
print("----------------------")
print("Optimizer: Adam")
print("Accuracy:", round(test_accuracy, 3))
print("F1:", round(test_f1, 3))


print("\nClassification report:")
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

confusion_df = pd.DataFrame(
    test_confusion_matrix,
    index=[
        "Actual public_transit",
        "Actual rideshare",
    ],
    columns=[
        "Predicted public_transit",
        "Predicted rideshare",
    ],
)

print("\nConfusion matrix:")
print(confusion_df)


# =============================================================================
# Save Model Output
# =============================================================================

test_output = test_df[
    [
        "q_transport_pref_text",
        "highest_transport",
    ]
].copy()

test_output["actual_binary"] = y_test
test_output["rideshare_probability"] = test_probabilities
test_output["predicted_binary"] = test_predictions

test_output["predicted_preference"] = (
    test_output["predicted_binary"]
    .map(
        {
            0: "public_transit",
            1: "rideshare",
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
            "optimizer": "Adam",
            "training_cases": len(train_df),
            "validation_cases": len(val_df),
            "test_cases": len(test_df),
            "epochs_trained": len(history.history["loss"]),
            "validation_accuracy": val_accuracy,
            "validation_f1": val_f1,
            "test_accuracy": test_accuracy,
            "validation_f1": val_f1,
        }
    ]
)

model_summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print("\nSaved test predictions to:")
print(PREDICTIONS_FILE)

print("\nSaved model summary to:")
print(SUMMARY_FILE)
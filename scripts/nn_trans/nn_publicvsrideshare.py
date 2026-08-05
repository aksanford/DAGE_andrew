"""
Neural network model for predicting participants' dominant transportation
preference from open-ended survey responses.

The script loads the cleaned transportation dataset, creates stratified
70/15/15 train-validation-test splits, vectorizes the text responses, compares
Adam, RMSprop, and SGD optimizers, evaluates the best model on the test set,
and saves an optimizer comparison table and figure.

Model architecture:
    Text input
    -> TextVectorization
    -> Embedding
    -> GlobalAveragePooling1D
    -> Dense ReLU layer
    -> Dropout
    -> Three-class softmax output

Evaluation metrics:
    - Classification accuracy
    - Macro-averaged F1 score
    - Per-class precision, recall, and F1 score

Reproducibility:
    Python, NumPy, and TensorFlow random seeds are set to 1738. The same seed
    is used when creating the train, validation, and test partitions.

Expected input:
    survey_transportation.csv

Required columns:
    q_transport_pref_text
        Participant's open-ended transportation preference response.

    highest_transport
        Participant's dominant transportation outcome category.

Outputs:
    optimizer_results.csv
        Validation accuracy, macro-F1, and training epochs for each optimizer.

    optimizer_comparison.png
        Bar chart comparing validation macro-F1 across optimizers.

Notes:
    The TextVectorization vocabulary is adapted using only the training data to
    prevent information leakage. The test set is reserved for final evaluation
    and is not used to select the optimizer.

Authors:
    DAGE research group

Date:
    July 2026
"""
 
 # =============================================================================
# Setup
# =============================================================================

# Imports

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score

from tensorflow import keras
from tensorflow.keras import layers


# Reproducible seed

SEED = 1738

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# File paths

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_clean"
OUT_DIR = PROJECT_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Load transportation dataset

transport = pd.read_csv(DATA_DIR / "survey_transportation.csv")

print(transport.head())

print(transport.columns)
print(transport.shape)
print(transport.isna().sum())


# Class distribution check
# Note: ties / no dominant transportation mode are treated as a third class.

transport["highest_transport"] = (
    transport["highest_transport"]
    .fillna("no_dominant_mode")
)

print("Outcome distribution:")
print(transport["highest_transport"].value_counts())

# Cases with no unique dominant transportation mode were coded as
# no_dominant_mode. This category includes both zero-use ties and non-zero ties
# between public transportation and rideshare use.


# =============================================================================
# Train / Validation / Test Split
# =============================================================================

# The dataset was partitioned using a stratified 70/15/15 split.
# Stratification preserves the relative frequency of the three transportation
# outcome classes across the training, validation, and test sets. A fixed
# random seed (1738) was used for reproducibility.

train_df, temp_df = train_test_split(
    transport,
    test_size=0.30,
    stratify=transport["highest_transport"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["highest_transport"],
    random_state=SEED
)


# Separate text predictors and outcome labels

X_train = train_df["q_transport_pref_text"].astype(str).to_numpy()
X_val = val_df["q_transport_pref_text"].astype(str).to_numpy()
X_test = test_df["q_transport_pref_text"].astype(str).to_numpy()

y_train_text = train_df["highest_transport"]
y_val_text = val_df["highest_transport"]
y_test_text = test_df["highest_transport"]


# Encode class labels as integers for the neural network

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train_text)
y_val = label_encoder.transform(y_val_text)
y_test = label_encoder.transform(y_test_text)

print("\nClass encoding:")
for number, label in enumerate(label_encoder.classes_):
    print(f"{number} = {label}")


# =============================================================================
# Text Vectorization
# =============================================================================

# Text vectorization adapted from proj_bible_example.

MAX_TOKENS = 5000
SEQUENCE_LENGTH = 100

vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)

# Learn vocabulary from training data only.

vectorizer.adapt(X_train)


# =============================================================================
# Build and Train Neural Network
# =============================================================================

def build_model(optimizer):
    model = keras.Sequential([
        layers.Input(shape=(1,), dtype=tf.string),

        vectorizer,

        layers.Embedding(
            input_dim=MAX_TOKENS,
            output_dim=64
        ),

        layers.GlobalAveragePooling1D(),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dropout(0.20),

        layers.Dense(
            3,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Optimizer candidates

optimizers = {
    "Adam": keras.optimizers.Adam(
        learning_rate=0.001
    ),
    "RMSprop": keras.optimizers.RMSprop(
        learning_rate=0.001
    ),
    "SGD": keras.optimizers.SGD(
        learning_rate=0.01
    )
}


# Early stopping: stop training when validation loss stops improving.

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# Compare optimizers using validation performance.

optimizer_results = []
trained_models = {}
training_histories = {}

for name, optimizer in optimizers.items():

    print(f"\nTraining with {name}")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = build_model(optimizer)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=0
    )

    val_probabilities = model.predict(
        X_val,
        verbose=0
    )

    val_predictions = np.argmax(
        val_probabilities,
        axis=1
    )

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro"
    )

    optimizer_results.append({
        "optimizer": name,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "epochs_trained": len(history.history["loss"])
    })

    trained_models[name] = model
    training_histories[name] = history


# Compare optimizer performance.

results_df = pd.DataFrame(
    optimizer_results
).sort_values(
    "validation_macro_f1",
    ascending=False
)

print("\nOptimizer comparison:")
print(results_df)

# Adam was the best-performing optimizer for this neural network configuration
# and validation split, with a macro-F1 of .427 and accuracy of .625.


# Save optimizer comparison outside Git-tracked data folders.

results_df.to_csv(
    OUT_DIR / "optimizer_results.csv",
    index=False
)


# Final test with best-performing optimizer.

best_optimizer_name = results_df.iloc[0]["optimizer"]
best_model = trained_models[best_optimizer_name]

test_probabilities = best_model.predict(
    X_test,
    verbose=0
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print("\nFINAL TEST PERFORMANCE")
print("Optimizer:", best_optimizer_name)
print("Accuracy:", round(test_accuracy, 3))
print("Macro F1:", round(test_macro_f1, 3))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# =============================================================================
# Visualizations
# =============================================================================

import matplotlib.pyplot as plt

plot_df = results_df.sort_values(
    "validation_macro_f1"
)

plt.figure(figsize=(7, 4))

plt.barh(
    plot_df["optimizer"],
    plot_df["validation_macro_f1"]
)

plt.xlabel("Validation Macro-F1")
plt.ylabel("Optimizer")
plt.title("Validation Performance by Optimizer")

plt.tight_layout()

plt.savefig(
    OUT_DIR / "optimizer_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print(f"Figure saved to {OUT_DIR / 'optimizer_comparison.png'}")
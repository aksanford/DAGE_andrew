import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 9499
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random
random.seed(SEED)

# File paths
PROJECT_DIR = Path("/Users/emilyhuffaker/Projects/DAGE/DAGE")
DATA_DIR = PROJECT_DIR / "data_clean"

# Load transportation dataset
transport = pd.read_csv(DATA_DIR / "survey_transportation.csv")

# Handle missing values in outcome
transport["highest_transport"] = (
    transport["highest_transport"]
    .fillna("no_dominant_mode")
)

# Train / Validation / Test Split (70/15/15 stratified)
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

# Encode class labels as integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_text)
y_val = label_encoder.transform(y_val_text)
y_test = label_encoder.transform(y_test_text)

# Text Vectorization
MAX_TOKENS = 5000
SEQUENCE_LENGTH = 100

vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)

# Learn vocabulary from training data only
vectorizer.adapt(X_train)

def build_model(hidden_activation, vectorizer, MAX_TOKENS):
    """Build neural network with specified activation function."""
    model_layers = [
        layers.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        layers.Embedding(input_dim=MAX_TOKENS, output_dim=64),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64)
    ]

    # Apply the selected hidden-layer activation function
    if hidden_activation == "Leaky ReLU":
        model_layers.append(layers.LeakyReLU(negative_slope=0.01))
    else:
        activation_lookup = {
            "ReLU": "relu",
            "tanh": "tanh",
            "GELU": "gelu"
        }
        model_layers.append(layers.Activation(activation_lookup[hidden_activation]))

    model_layers.extend([
        layers.Dropout(0.20),
        layers.Dense(3, activation="softmax")  # 3-class outcome
    ])

    model = keras.Sequential(model_layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def test_activation_functions():
    """Test different activation functions on the neural network."""
    activation_functions = ["ReLU", "tanh", "Leaky ReLU", "GELU"]
    activation_results = []
    trained_activation_models = {}

    for activation_name in activation_functions:
        print(f"\nTraining with {activation_name}")

        # Reset seed for reproducibility
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        model = build_model(activation_name, vectorizer, MAX_TOKENS)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=0
        )

        val_probabilities = model.predict(X_val, verbose=0)
        val_predictions = np.argmax(val_probabilities, axis=1)

        val_accuracy = accuracy_score(y_val, val_predictions)
        val_macro_f1 = f1_score(y_val, val_predictions, average="macro")

        activation_results.append({
            "activation": activation_name,
            "validation_macro_f1": val_macro_f1,
            "validation_accuracy": val_accuracy,
            "epochs_trained": len(history.history["loss"])
        })

        trained_activation_models[activation_name] = model

    return activation_results, trained_activation_models


def plot_results(activation_results):
    """Plot comparison of activation function performance."""
    activation_results_df = pd.DataFrame(activation_results).sort_values(
        "validation_macro_f1", ascending=False
    )

    print("\nActivation function comparison:")
    print(activation_results_df)

    plot_df = activation_results_df.sort_values("validation_macro_f1")

    plt.figure(figsize=(7, 4))
    plt.barh(plot_df["activation"], plot_df["validation_macro_f1"])
    plt.xlabel("Validation Macro-F1")
    plt.ylabel("Hidden-Layer Activation Function")
    plt.title("Validation Performance by Activation Function")
    plt.tight_layout()
    plt.savefig("activation_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    results, models = test_activation_functions()
    plot_results(results)
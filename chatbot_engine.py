"""
chatbot_engine.py — Inference Engine for the Deep Learning Chatbot
===================================================================
Supports both:
  1. Pure NumPy forward pass using exported weights (fast, <1MB, serverless & Vercel compatible)
  2. Keras model fallback (when TensorFlow is installed)
"""

import os
import json
import pickle
import random
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------
# Safe NLTK Setup (works on Vercel / read-only serverless filesystems)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nltk_data_dir = os.path.join("/tmp", "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
    nltk.download("wordnet", download_dir=nltk_data_dir, quiet=True)
    nltk.download("omw-1.4", download_dir=nltk_data_dir, quiet=True)
    nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
except Exception:
    pass

lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------------
# Load Artefacts
# ------------------------------------------------------------------
words_path = os.path.join(BASE_DIR, "model", "words.pkl")
classes_path = os.path.join(BASE_DIR, "model", "classes.pkl")
intents_path = os.path.join(BASE_DIR, "intents.json")
npz_path = os.path.join(BASE_DIR, "model", "model_weights.npz")
keras_path = os.path.join(BASE_DIR, "model", "chatbot_model.keras")

words = pickle.load(open(words_path, "rb"))
classes = pickle.load(open(classes_path, "rb"))

with open(intents_path, "r") as f:
    intents_data = json.load(f)

# Load weights for NumPy forward pass or load Keras model
npz_weights = None
keras_model = None

if os.path.exists(npz_path):
    npz_weights = np.load(npz_path)
elif os.path.exists(keras_path):
    try:
        from tensorflow.keras.models import load_model
        keras_model = load_model(keras_path)
    except Exception as e:
        print(f"Warning: Could not load Keras model: {e}")

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def tokenize_and_lemmatize(sentence: str) -> list[str]:
    """Tokenise and lemmatise a sentence with zero-dependency fallback."""
    try:
        tokens = nltk.word_tokenize(sentence)
    except Exception:
        tokens = sentence.replace("?", " ").replace("!", " ").replace(".", " ").replace(",", " ").split()

    results = []
    for t in tokens:
        w = t.lower().strip()
        if not w:
            continue
        try:
            results.append(lemmatizer.lemmatize(w))
        except Exception:
            results.append(w)
    return results


def bag_of_words(sentence: str) -> np.ndarray:
    """Convert sentence to a BoW vector over the training vocabulary."""
    token_words = tokenize_and_lemmatize(sentence)
    bag = [1 if w in token_words else 0 for w in words]
    return np.array(bag, dtype=np.float32)


# ------------------------------------------------------------------
# Deep Learning Forward Propagation (Pure NumPy or Keras)
# ------------------------------------------------------------------
def numpy_forward(bow: np.ndarray) -> np.ndarray:
    """
    Computes exact forward pass of the trained 3-layer ANN:
      Dense(256) -> BatchNorm -> ReLU
      Dense(128) -> BatchNorm -> ReLU
      Dense(64)  -> ReLU
      Dense(num_classes) -> Softmax
    """
    w = npz_weights
    x = np.array([bow], dtype=np.float32)

    # Layer 0: Dense(256)
    x = np.maximum(0, np.dot(x, w['layer_0_0']) + w['layer_0_1'])
    # Layer 1: BatchNorm (gamma, beta, mean, var)
    gamma, beta, mean, var = w['layer_1_0'], w['layer_1_1'], w['layer_1_2'], w['layer_1_3']
    x = gamma * (x - mean) / np.sqrt(var + 1e-3) + beta

    # Layer 3: Dense(128)
    x = np.maximum(0, np.dot(x, w['layer_3_0']) + w['layer_3_1'])
    # Layer 4: BatchNorm
    gamma, beta, mean, var = w['layer_4_0'], w['layer_4_1'], w['layer_4_2'], w['layer_4_3']
    x = gamma * (x - mean) / np.sqrt(var + 1e-3) + beta

    # Layer 6: Dense(64)
    x = np.maximum(0, np.dot(x, w['layer_6_0']) + w['layer_6_1'])

    # Layer 8: Dense(output) + Softmax
    logits = np.dot(x, w['layer_8_0']) + w['layer_8_1']
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs[0]


ERROR_THRESHOLD = 0.25


def predict_intent(sentence: str) -> list[dict]:
    """Returns sorted intent predictions above ERROR_THRESHOLD."""
    bow = bag_of_words(sentence)

    if npz_weights is not None:
        predictions = numpy_forward(bow)
    elif keras_model is not None:
        predictions = keras_model.predict(np.array([bow]), verbose=0)[0]
    else:
        raise RuntimeError("No model weights or Keras model found.")

    results = [
        {"intent": classes[i], "probability": float(p)}
        for i, p in enumerate(predictions)
        if p > ERROR_THRESHOLD
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def get_response(sentence: str) -> dict:
    """Main entry point: predict intent -> pick response."""
    intents_list = predict_intent(sentence)

    if not intents_list:
        tag = "no_response"
        confidence = 0.0
    else:
        tag = intents_list[0]["intent"]
        confidence = intents_list[0]["probability"]

    for intent in intents_data["intents"]:
        if intent["tag"] == tag and intent["responses"]:
            response = random.choice(intent["responses"])
            return {
                "intent": tag,
                "confidence": round(confidence * 100, 1),
                "response": response,
            }

    return {
        "intent": "unknown",
        "confidence": 0.0,
        "response": "I'm not sure how to answer that. Could you rephrase?",
    }

"""
chatbot_engine.py — Inference Engine for the Deep Learning Chatbot
===================================================================
Loads the pre-trained Keras model and provides:
  - predict_intent(text): returns the predicted intent tag + confidence
  - get_response(text):   returns a random response for the predicted intent
"""

import json
import pickle
import random
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------
nltk.download("punkt",     quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",  quiet=True)
nltk.download("punkt_tab", quiet=True)

lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------------
# Load artefacts
# ------------------------------------------------------------------
words   = pickle.load(open("model/words.pkl",   "rb"))
classes = pickle.load(open("model/classes.pkl", "rb"))
model   = load_model("model/chatbot_model.keras")

with open("intents.json", "r") as f:
    intents_data = json.load(f)

# ------------------------------------------------------------------
# Preprocessing helpers
# ------------------------------------------------------------------
def tokenize_and_lemmatize(sentence: str) -> list[str]:
    """Tokenise and lemmatise a sentence."""
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(t.lower()) for t in tokens]


def bag_of_words(sentence: str) -> np.ndarray:
    """Convert sentence to a BoW vector over the training vocabulary."""
    token_words = tokenize_and_lemmatize(sentence)
    bag = [1 if w in token_words else 0 for w in words]
    return np.array(bag, dtype=np.float32)


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
ERROR_THRESHOLD = 0.25


def predict_intent(sentence: str) -> list[dict]:
    """
    Returns a list of dicts sorted by probability (descending).
    Each dict: {"intent": str, "probability": float}
    """
    bow = bag_of_words(sentence)
    predictions = model.predict(np.array([bow]), verbose=0)[0]

    results = [
        {"intent": classes[i], "probability": float(p)}
        for i, p in enumerate(predictions)
        if p > ERROR_THRESHOLD
    ]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def get_response(sentence: str) -> dict:
    """
    Main entry point: predict intent → pick a random matching response.
    Returns: {intent, confidence, response}
    """
    intents_list = predict_intent(sentence)

    if not intents_list:
        # Fallback to no_response intent
        tag = "no_response"
        confidence = 0.0
    else:
        tag        = intents_list[0]["intent"]
        confidence = intents_list[0]["probability"]

    # Find matching responses
    for intent in intents_data["intents"]:
        if intent["tag"] == tag and intent["responses"]:
            response = random.choice(intent["responses"])
            return {
                "intent":     tag,
                "confidence": round(confidence * 100, 1),
                "response":   response,
            }

    return {
        "intent":     "unknown",
        "confidence": 0.0,
        "response":   "I'm not sure how to answer that. Could you rephrase?",
    }


# ------------------------------------------------------------------
# Quick CLI test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Chatbot Engine — type 'quit' to exit\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in ("quit", "exit"):
            break
        result = get_response(msg)
        print(f"Bot [{result['intent']} | {result['confidence']}%]: {result['response']}\n")

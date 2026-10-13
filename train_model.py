"""
train_model.py — Deep Learning Chatbot Intent Classifier
========================================================
Architecture:
  - Input: Bag-of-Words vector from tokenized/lemmatized user text
  - Hidden Layer 1: Dense(256) + ReLU + Dropout(0.5)
  - Hidden Layer 2: Dense(128) + ReLU + Dropout(0.5)
  - Output: Dense(num_classes) + Softmax → predicted intent

Steps:
  1. Load intents.json
  2. Tokenize & lemmatize all patterns
  3. Build vocabulary (words) and class list (tags)
  4. Convert to Bag-of-Words training samples
  5. Train a Sequential Neural Network (TensorFlow/Keras)
  6. Save model + preprocessed data to disk
"""

import json
import pickle
import random
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------------------------------------------
# 1. Download NLTK data
# ------------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt_tab", quiet=True)

lemmatizer = WordNetLemmatizer()

# ------------------------------------------------------------------
# 2. Load intents
# ------------------------------------------------------------------
with open("intents.json", "r") as f:
    intents_data = json.load(f)

words = []       # vocabulary
classes = []     # unique intent tags
documents = []   # (tokenized_pattern, tag) pairs
ignore_chars = ["?", "!", ".", ",", "'", "-", ";", ":"]

# ------------------------------------------------------------------
# 3. Tokenise & lemmatise all patterns
# ------------------------------------------------------------------
for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"  Vocabulary size : {len(words)}")
print(f"  Intent classes  : {len(classes)}")
print(f"  Training samples: {len(documents)}")

# Save vocabulary & classes for inference
pickle.dump(words,   open("model/words.pkl",   "wb"))
pickle.dump(classes, open("model/classes.pkl", "wb"))

# ------------------------------------------------------------------
# 4. Build Bag-of-Words training set
# ------------------------------------------------------------------
training = []
output_empty = [0] * len(classes)

for doc_words, tag in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc_words]
    for w in words:
        bag.append(1 if w in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]), dtype=np.float32)
train_y = np.array(list(training[:, 1]), dtype=np.float32)

print(f"  train_x shape: {train_x.shape}")
print(f"  train_y shape: {train_y.shape}")

# ------------------------------------------------------------------
# 5. Build the Neural Network
# ------------------------------------------------------------------
model = Sequential([
    Dense(256, input_shape=(len(train_x[0]),), activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(len(train_y[0]), activation="softmax"),
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)
model.summary()

# ------------------------------------------------------------------
# 6. Train
# ------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="loss", patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1),
]

hist = model.fit(
    train_x, train_y,
    epochs=300,
    batch_size=8,
    verbose=1,
    callbacks=callbacks,
)

# ------------------------------------------------------------------
# 7. Save model
# ------------------------------------------------------------------
model.save("model/chatbot_model.keras")

final_acc  = hist.history["accuracy"][-1]
final_loss = hist.history["loss"][-1]
print(f"\n✅ Training complete!")
print(f"   Final accuracy : {final_acc*100:.2f}%")
print(f"   Final loss     : {final_loss:.4f}")
print("   Model saved to model/chatbot_model.keras")

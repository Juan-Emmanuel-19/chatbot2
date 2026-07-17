import random
import json
import pickle
import numpy as np
import nltk
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras import Input

# ==========================================
# DESCARGA DE RECURSOS NLTK
# ==========================================

nltk.download("punkt_tab")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# ==========================================
# CARGAR INTENTS
# ==========================================

with open("intents.json", encoding="utf-8") as archivo:
    intents = json.load(archivo)

words = []
classes = []
documents = []

ignore_letters = ["?", "!", "¿", "¡", ".", ","]

# ==========================================
# PROCESAMIENTO DEL JSON
# ==========================================

for intent in intents["intents"]:

    for pattern in intent["patterns"]:

        word_list = nltk.word_tokenize(pattern)

        words.extend(word_list)

        documents.append((word_list, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_letters
]

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# ==========================================
# BAG OF WORDS
# ==========================================

training = []

output_empty = [0] * len(classes)

for document in documents:

    bag = []

    word_patterns = [
        lemmatizer.lemmatize(word.lower())
        for word in document[0]
    ]

    for word in words:

        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)

    output_row[
        classes.index(document[1])
    ] = 1

    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([i[0] for i in training])

train_y = np.array([i[1] for i in training])

# ==========================================
# MODELO
# ==========================================

model = Sequential([

    Input(shape=(len(train_x[0]),)),

    Dense(256, activation="relu"),
    Dropout(0.4),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(len(train_y[0]), activation="softmax")

])

# ==========================================
# COMPILACIÓN
# ==========================================

sgd = SGD(

    learning_rate=0.001,
    momentum=0.9,
    nesterov=True

)

model.compile(

    loss="categorical_crossentropy",
    optimizer=sgd,
    metrics=["accuracy"]

)

# ==========================================
# ENTRENAMIENTO
# ==========================================

print("\n===================================")
print("Entrenando Aurora...")
print("===================================\n")

hist = model.fit(

    train_x,
    train_y,
    epochs=300,
    batch_size=8,
    verbose=1

)

# ==========================================
# EVALUACIÓN
# ==========================================

loss, accuracy = model.evaluate(

    train_x,
    train_y,
    verbose=0

)

print("\n===================================")
print("RESULTADOS DEL ENTRENAMIENTO")
print("===================================")

print(f"Accuracy final : {accuracy*100:.2f}%")
print(f"Loss final     : {loss:.4f}")

# ==========================================
# GUARDAR MODELO
# ==========================================

model.save("chatbot_model.h5")

print("\nModelo guardado correctamente.")

# ==========================================
# GUARDAR GRÁFICA
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(hist.history["accuracy"])

plt.title("Accuracy del entrenamiento")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig("accuracy.png")

print("Gráfica guardada como accuracy.png") 
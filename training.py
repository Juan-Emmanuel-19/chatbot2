import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Librerías actualizadas para el modelo neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Descargas de NLTK (con la corrección de _tab)
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Cargamos el archivo JSON (especificando utf-8 para no tener problemas con acentos)
intents = json.loads(open('intents.json', encoding='utf-8').read())

words = []
classes = []
documents = []
# Símbolos que la IA debe ignorar al aprender
ignore_letters = ['?', '!', '¿', '¡', '.', ',']

# 1. CLASIFICACIÓN DE PATRONES Y CATEGORÍAS
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Separamos cada frase en palabras (tokenización)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Relacionamos las palabras con su etiqueta (tag)
        documents.append((word_list, intent['tag']))
        # Si la etiqueta no está en nuestra lista de clases, la añadimos
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# 2. LEMATIZACIÓN Y LIMPIEZA
# Convertimos las palabras a su raíz minúscula y quitamos los símbolos
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]

# Eliminamos duplicados y ordenamos
words = sorted(set(words))
classes = sorted(set(classes))

# Guardamos los datos procesados para que el chatbot los use después
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# 3. PREPARACIÓN DE LOS DATOS DE ENTRENAMIENTO
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Creamos nuestra "bolsa de palabras" (1 si la palabra está, 0 si no)
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclamos los datos para que el entrenamiento sea aleatorio
random.shuffle(training)

# Separamos los datos de entrada (x) y salida (y) de forma segura para NumPy moderno
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# 4. CREACIÓN DE LA RED NEURONAL (MODELO)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# 5. COMPILACIÓN Y ENTRENAMIENTO
# Usamos el optimizador estándar SGD (el que corregimos en la línea 11)
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenamos el modelo. 
# epochs = cantidad de veces que la red verá los datos.
print("Entrenando el modelo...")
hist = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)

# Guardamos el cerebro de la IA
model.save('modelo_mision_colombia.h5', hist)
print("¡Entrenamiento terminado y modelo guardado con éxito!")
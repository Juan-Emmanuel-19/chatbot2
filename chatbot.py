import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from spellchecker import SpellChecker

# Inicializamos el lematizador y el corrector ortográfico en español
lemmatizer = WordNetLemmatizer()
corrector = SpellChecker(language='es')

# 1. CARGAMOS LOS ARCHIVOS GENERADOS EN EL ENTRENAMIENTO
# Especificamos encoding='utf-8' para evitar problemas con tildes y eñes
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# 2. FUNCIONES DE PROCESAMIENTO DE TEXTO
def clean_up_sentence(sentence):
    """Separa la oración del usuario, corrige la ortografía y la simplifica (lematiza)"""
    sentence_words = nltk.word_tokenize(sentence)
    
    palabras_corregidas = []
    for word in sentence_words:
        # Intentamos corregir la palabra (ej: 'burnas' -> 'buenas')
        palabra_bien_escrita = corrector.correction(word)
        
        # Si el corrector devuelve None (no sabe qué hacer), usamos la original
        if palabra_bien_escrita is None:
            palabra_bien_escrita = word
            
        # Lematizamos la palabra ya corregida y la pasamos a minúsculas
        palabras_corregidas.append(lemmatizer.lemmatize(palabra_bien_escrita.lower()))
        
    return palabras_corregidas

def bag_of_words(sentence):
    """Convierte la oración del usuario en una matriz de 0s y 1s para la red neuronal"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# 3. PREDICCIÓN Y RESPUESTA
def predict_class(sentence):
    """Pasa la matriz por la red neuronal y devuelve las categorías con mayor probabilidad"""
    bow = bag_of_words(sentence)
    # model.predict espera un arreglo 2D, por eso usamos np.array([bow])
    res = model.predict(np.array([bow]), verbose=0)[0] # verbose=0 oculta los logs de predicción en consola
    
    # Filtramos las predicciones que tengan muy baja probabilidad (umbral del 25%)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Ordenamos de mayor a menor probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """Busca la etiqueta predicha en el JSON y elige una respuesta aleatoria"""
    # Si la lista de intenciones está vacía (no superó el umbral del 25%), damos una respuesta por defecto
    if not intents_list:
        return "Lo siento, no te he entendido muy bien. ¿Podrías escribirlo de otra forma?"
        
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            # Selecciona una respuesta al azar de las opciones disponibles en esa categoría
            result = random.choice(i['responses'])
            break
    return result

# 4. BUCLE PRINCIPAL DE INTERACCIÓN (EL CHAT)
print("-" * 50)
print("Aquí estoy para ayudarte, cuenta conmigo, ¿cuál es tu situación?")
print("-" * 50)

while True:
    message = input("Tú: ")
    
    if message.lower() == "salir":
        print("Bot: ¡Nos vemos pronto!")
        break
        
    # Predecimos la intención y obtenemos la respuesta
    ints = predict_class(message)
    res = get_response(ints, intents)
    
    print("Bot:", res)
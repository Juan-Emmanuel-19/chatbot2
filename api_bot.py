print("ESTE ES MI API_BOT NUEVO")
from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

import os
import tensorflow as tf


import os

print("Archivos en la carpeta:")
print(os.listdir("."))

# --- 1. DESCARGAS Y PREPARACIÓN DE NLTK ---


nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# --- 2. INICIALIZACIÓN DE LA IA ---
lemmatizer = WordNetLemmatizer()
corrector = SpellChecker(language='es')

# Carga de los archivos entrenados
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5', compile=False)

# --- 3. FUNCIONES DEL CEREBRO (NLP) ---
def clean_up_sentence(sentence):
    """Limpia y corrige ortografía del mensaje"""
    sentence_words = nltk.word_tokenize(sentence)
    palabras_corregidas = []
    for word in sentence_words:
        palabra_bien_escrita = corrector.correction(word)
        if palabra_bien_escrita is None:
            palabra_bien_escrita = word
        palabras_corregidas.append(lemmatizer.lemmatize(palabra_bien_escrita.lower()))
    return palabras_corregidas

def bow(sentence, words, show_details=True):
    """Convierte el texto a una bolsa de palabras (matriz numérica)"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    """Predice a qué categoría (tag) pertenece el mensaje"""
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    """Obtiene la respuesta redactada desde el JSON"""
    if not ints:
        return "Lo siento, no he comprendido bien tu mensaje. ¿Podrías explicarme un poco más?"
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# --- 4. CONFIGURACIÓN DEL SERVIDOR WEB (FLASK API) ---
print("App creada")
app = Flask(__name__)

@app.route("/")


def inicio():
    return "Aurora API funcionando correctamente"


@app.route("/prueba", methods=["POST"])
def prueba():
    return jsonify({
        "estado": "ok",
        "mensaje": "La API recibe POST correctamente"
    })
    
@app.route('/enviar_mensaje', methods=['POST'])
def procesar_mensaje():
    # Recibimos el JSON desde la App en Flutter
    datos_entrantes = request.get_json(force=True, silent=True)
    
    if not datos_entrantes or 'mensaje' not in datos_entrantes:
        return jsonify({"error": "Falta el parámetro 'mensaje' en el JSON"}), 400

    texto_usuario = datos_entrantes['mensaje']
    
    # --- 4.1 LÓGICA DE BIENVENIDA (Protocolo del psicólogo) ---
    if texto_usuario == "iniciar_sesion":
        return jsonify({
            "nivel_riesgo": "neutro",
            "fase": "inicio_o_cierre",
            "respuesta_bot": "¡Hola! Soy Aurora. Antes de empezar, quiero recordarte que este es un espacio confidencial y seguro. Si en algún momento sientes que tu integridad corre peligro, recuerda que puedes presionar el botón de salida rápida. ¿Cómo te sientes hoy y en qué te puedo acompañar?",
            "bloquear_entrada_texto": False,
            "botones_activos": []
        }), 200

    try:
        # La IA analiza el texto
        ints = predict_class(texto_usuario, model)
        respuesta_ia = get_response(ints, intents)
        
        # Extraemos la etiqueta (tag) ganadora y su probabilidad
        if len(ints) > 0:
            intencion_detectada = ints[0]['intent']
            probabilidad = float(ints[0]['probability'])
        else:
            intencion_detectada = "desconocida"
            probabilidad = 0.0

        # --- 5. LÓGICA DE TRIAGE (El diagrama cobrando vida) ---
        
        # A) NIVEL ROJO (Peligro Inminente)
        if intencion_detectada == "sos_inminente":
            payload = {
                "nivel_riesgo": "rojo",
                "fase": "1_override_emergencia",
                "respuesta_bot": respuesta_ia,  
                "bloquear_entrada_texto": True, 
                "botones_activos": [
                    {"label": "Llamar Policía (123)", "accion": "tel:123"},
                    {"label": "Línea Mujer (155)", "accion": "tel:155"}
                ]
            }

        # B) NIVEL NARANJA (Riesgo de Escalamiento)
        elif intencion_detectada == "acoso_amenaza_incipiente":
            payload = {
                "nivel_riesgo": "naranja",
                "fase": "2_modo_contencion",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": False,
                "botones_activos": [
                    {"label": "Diseñar Plan de Seguridad", "accion": "enviar_mensaje: quiero un plan de seguridad"},
                    {"label": "Ruta Universitaria", "accion": "enviar_mensaje: quiero conocer las rutas"}
                ]
            }

        # C) NIVEL NEUTRO (Saludos)
        elif intencion_detectada == "saludo":
            payload = {
                "nivel_riesgo": "neutro",
                "fase": "inicio_o_cierre",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": False,
                "botones_activos": []
            }

        # D) NIVEL NEUTRO (Despedidas y Cierre de conversación)
        elif intencion_detectada == "despedida":
            payload = {
                "nivel_riesgo": "neutro",
                "fase": "inicio_o_cierre",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": False,
                "botones_activos": [
                    {"label": "Finalizar sesión", "accion": "cerrar_app"},
                    {"label": "No, quiero seguir hablando", "accion": "enviar_mensaje: Hola"}
                ]
            }

        # E) NIVEL AMARILLO (Información, Dudas, Marco Legal, Consecuencias)
        else:
            payload = {
                "nivel_riesgo": "amarillo",
                "fase": "3_modo_psicoeducativo",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": False,
                "botones_activos": [
                    {"label": "Señales de Alerta", "accion": "enviar_mensaje: ¿Cómo identificar señales de alerta?"},
                    {"label": "Marco Legal", "accion": "enviar_mensaje: ¿Qué leyes protegen a la mujer?"},
                    {"label": "Efectos Emocionales", "accion": "enviar_mensaje: Consecuencias de sufrir VSBG"}
                ]
            }

        # Devolvemos el paquete estructurado a Flutter
        return jsonify(payload), 200

    except Exception as e:
        print(f"❌ Error en el procesamiento del bot: {e}")
        return jsonify({
            "nivel_riesgo": "neutro",
            "fase": "error_interno",
            "respuesta_bot": "Lo siento, tuve un pequeño problema técnico al procesar tu mensaje. ¿Podrías intentarlo de nuevo?",
            "bloquear_entrada_texto": False,
            "botones_activos": []
        }), 500

# Punto de arranque del servidor

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# ---------------------------------------------------------
# CÓDIGO DE RESPALDO PARA FLUTTER (DART)
# ---------------------------------------------------------
# Future<void> enviarMensaje(String mensaje) async {
#   final response = await http.post(
#     Uri.parse('http://192.168.1.XX:5000/enviar_mensaje'), // <-- LA DIRECCIÓN DE TU LAP
#     headers: {"Content-Type": "application/json"},
#     body: jsonEncode({"mensaje": mensaje}), // <-- EL MENSAJE
#   );
#
#   if (response.statusCode == 200) {
#     // AQUÍ ÉL RECIBE EL JSON QUE TU PYTHON LE MANDÓ
#     var datos = jsonDecode(response.body);
#     print("El bot respondió: ${datos['respuesta_bot']}");
#   }
# }
#
# @override
# void initState() {
#   super.initState();
#   // Esta línea le dice a tu Python: "Oye, ya se abrió el chat, dame el mensaje inicial"
#   enviarMensaje("iniciar_sesion"); 
# } 
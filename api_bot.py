print("ESTE ES MI API_BOT NUEVO")

from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
import os
import tensorflow as tf

from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

print("Archivos en la carpeta:")
print(os.listdir("."))

# ===========================
# DESCARGAS NLTK
# ===========================

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ===========================
# CARGA DEL MODELO
# ===========================

lemmatizer = WordNetLemmatizer()
corrector = SpellChecker(language="es")

intents = json.loads(open("intents.json", encoding="utf-8").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = tf.keras.models.load_model("chatbot_model.h5", compile=False)

# ===========================
# FUNCIONES NLP
# ===========================

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)

    palabras_corregidas = []

    for word in sentence_words:
        palabra = corrector.correction(word)

        if palabra is None:
            palabra = word

        palabras_corregidas.append(
            lemmatizer.lemmatize(palabra.lower())
        )

    return palabras_corregidas


def bow(sentence, words, show_details=True):

    print("Entró a bow")

    sentence_words = clean_up_sentence(sentence)

    print(sentence_words)

    ...

    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence, model):

    print("1. Entró a predict_class")

    p = bow(sentence, words, show_details=False)

    print("2. Bag creado")

    res = model.predict(np.array([p]))[0]

    print("3. Predicción realizada")

    ERROR_THRESHOLD = 0.25

    ...
    results = [
        [i, r]
        for i, r in enumerate(res)
        if r > ERROR_THRESHOLD
    ]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:

        return_list.append({
            "intent": classes[r[0]],
            "probability": str(r[1])
        })

    return return_list


def get_response(ints, intents_json):

    if len(ints) == 0:
        return "Lo siento, no entendí tu mensaje."

    tag = ints[0]["intent"]

    for intent in intents_json["intents"]:

        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Lo siento, no encontré una respuesta."

# ===========================
# FLASK
# ===========================

app = Flask(__name__)

print("App creada")

@app.route("/")
def inicio():

    return "Aurora API funcionando correctamente"


@app.route("/prueba", methods=["POST"])
def prueba():

    return jsonify({
        "estado": "ok",
        "mensaje": "La API recibe POST correctamente"
    })

print("===== API CARGADA =====")
@app.route("/enviar_mensaje", methods=["POST"])
def procesar_mensaje():

    print("===== ENTRÓ A procesar_mensaje =====")
    datos_entrantes = request.get_json(force=True, silent=True)
    
    if not datos_entrantes:

        return jsonify({
            "error": "No se recibió JSON"
        }), 400

    if "mensaje" not in datos_entrantes:

        return jsonify({
            "error": "Falta el parámetro mensaje"
        }), 400

    texto_usuario = datos_entrantes["mensaje"]
    
    print("========== INICIO MENSAJE ==========")
    print("Mensaje recibido:", texto_usuario)

    if texto_usuario == "iniciar_sesion":

        return jsonify({

            "nivel_riesgo": "neutro",

            "fase": "inicio_o_cierre",

            "respuesta_bot":
            "¡Hola! Soy Aurora. Antes de empezar, quiero recordarte que este es un espacio confidencial y seguro. Si en algún momento sientes que tu integridad corre peligro, recuerda que puedes presionar el botón de salida rápida. ¿Cómo te sientes hoy y en qué te puedo acompañar?",

            "bloquear_entrada_texto": False,

            "botones_activos": []

        }), 200


    try:
        

        ints = predict_class(texto_usuario, model)
        respuesta_ia = get_response(ints, intents)
    

        if len(ints) > 0:
            intencion_detectada = ints[0]["intent"]
            probabilidad = float(ints[0]["probability"])
        else:
            intencion_detectada = "desconocida"
            probabilidad = 0.0

        print("Intent:", intencion_detectada)
        print("Probabilidad:", probabilidad)

        # ==========================
        # NIVEL ROJO
        # ==========================

        if intencion_detectada == "sos_inminente":

            payload = {
                "nivel_riesgo": "rojo",
                "fase": "1_override_emergencia",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": True,
                "botones_activos": [
                    {
                        "label": "Llamar Policía (123)",
                        "accion": "tel:123"
                    },
                    {
                        "label": "Línea Mujer (155)",
                        "accion": "tel:155"
                    }
                ]
            }

        # ==========================
        # NIVEL NARANJA
        # ==========================

        elif intencion_detectada == "acoso_amenaza_incipiente":

            payload = {
                "nivel_riesgo": "naranja",
                "fase": "2_modo_contencion",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": False,
                "botones_activos": [
                    {
                        "label": "Diseñar Plan de Seguridad",
                        "accion": "enviar_mensaje: quiero un plan de seguridad"
                    },
                    {
                        "label": "Ruta Universitaria",
                        "accion": "enviar_mensaje: quiero conocer las rutas"
                    }
                ]
            }

        # ==========================
        # SALUDO
        # ==========================

        elif intencion_detectada == "saludo":

            payload = {
                "nivel_riesgo": "neutro",
                "fase": "inicio_o_cierre",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": False,
                "botones_activos": []
            }

        # ==========================
        # DESPEDIDA
        # ==========================

        elif intencion_detectada == "despedida":

            payload = {
                "nivel_riesgo": "neutro",
                "fase": "inicio_o_cierre",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": False,
                "botones_activos": [
                    {
                        "label": "Finalizar sesión",
                        "accion": "cerrar_app"
                    },
                    {
                        "label": "No, quiero seguir hablando",
                        "accion": "enviar_mensaje: Hola"
                    }
                ]
            }

        # ==========================
        # CUALQUIER OTRA CONSULTA
        # ==========================

        else:

            payload = {
                "nivel_riesgo": "amarillo",
                "fase": "3_modo_psicoeducativo",
                "respuesta_bot": respuesta_ia,
                "bloquear_entrada_texto": False,
                "botones_activos": [
                    {
                        "label": "Señales de Alerta",
                        "accion": "enviar_mensaje: ¿Cómo identificar señales de alerta?"
                    },
                    {
                        "label": "Marco Legal",
                        "accion": "enviar_mensaje: ¿Qué leyes protegen a la mujer?"
                    },
                    {
                        "label": "Efectos Emocionales",
                        "accion": "enviar_mensaje: Consecuencias de sufrir VSBG"
                    }
                ]
            }

        return jsonify(payload), 200
    
    except Exception as e:
        import traceback

        print("=" * 60)
        print("❌ ERROR EN procesar_mensaje()")
        traceback.print_exc()
        print("=" * 60)

        return jsonify({
            "nivel_riesgo": "neutro",
            "fase": "error_interno",
            "respuesta_bot": "Lo siento, ocurrió un error interno.",
            "error": str(e),
            "bloquear_entrada_texto": False,
            "botones_activos": []
        }), 500


# ======================================================
# INICIO DEL SERVIDOR
# ======================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port
    )


# ======================================================
# EJEMPLO DE CONSUMO DESDE FLUTTER
# ======================================================

#
#Future<void> enviarMensaje(String mensaje) async {

 # final response = await http.post(
  #  Uri.parse(
   #   'https://TU-API.onrender.com/enviar_mensaje',
    #),
    #headers: {
    #  "Content-Type": "application/json"
    #},
    #body: jsonEncode({
     # "mensaje": mensaje
    #}),
  #);

  #if (response.statusCode == 200) {

   # final datos = jsonDecode(response.body);

    #print(datos["respuesta_bot"]);

  #} else {

   # print(response.body);

  #}

#}
""
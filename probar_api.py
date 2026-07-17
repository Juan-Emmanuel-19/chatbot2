import requests

url = "https://api-bot-flutter.onrender.com/enviar_mensaje"

datos = {
    "mensaje": "¿Como puedo hacer una denuncia?"
}

r = requests.post(url, json=datos)

print(r.status_code)
print(r.text)
import tensorflow as tf
from tensorflow.keras.models import load_model

# Luego intenta cargar el modelo
model = load_model('chatbot_model.h5')

# 2. Lo guardas de forma limpia y sin optimizador
model.save('chatbot_model_limpio.h5', include_optimizer=False)

print("¡Modelo guardado como 'chatbot_model_limpio.h5'!")   
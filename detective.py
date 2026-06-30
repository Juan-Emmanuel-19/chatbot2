import pickle
import json
import os

print("\n" + "="*55)
print("🕵️‍♂️ INFORME FORENSE DEL CEREBRO DE LA IA 🕵️‍♂️")
print("📍 Carpeta donde estoy parado:", os.getcwd())

try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
        
    tags_en_json = [item['tag'] for item in intents['intents']]

    print(f"\n1. Total de palabras en vocabulario: {len(words)}")
    print(f"2. ¿La IA conoce la palabra 'matar'?: {'matar' in words}")
    print(f"3. Total de etiquetas en su memoria: {len(classes)}")
    print(f"4. ¿La etiqueta 'sos_inminente' está en el Pickle?: {'sos_inminente' in classes}")
    print(f"5. ¿La etiqueta 'sos_inminente' está en el JSON?: {'sos_inminente' in tags_en_json}")

except Exception as e:
    print(f"\n❌ ERROR FATAL LEYENDO ARCHIVOS: {e}")

print("="*55 + "\n")
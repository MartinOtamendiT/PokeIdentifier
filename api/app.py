from flask import Flask, request, render_template, session, jsonify, redirect, url_for,current_app, g, flash
from flask_cors import CORS
import pickle
#from pokemon_identifier import identify_pokemon, POKEMON_DATASET
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import io
import base64

# Carga el modelo y el procesador CLIP.
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
# Se asegura de usar GPU si está disponible.
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Carga el dataset de embeddings de Pokémon.
try:
    POKEMON_DATASET = np.load("pokemon_embeddings_dataset.npy", allow_pickle=True).item()
    print("Dataset de Pokémon cargado exitosamente.")
except FileNotFoundError:
    print("Error: 'pokemon_embeddings.npy' no encontrado. Por favor, ejecuta 'embeddings_dataset_generator.py' primero.")
    POKEMON_DATASET = {} # Inicializar vacío para evitar errores

#Configuración de la aplicación Flask
app = Flask(__name__)
CORS(app) #Esto habilita CORS para todas las rutas y orígenes.

# Función para generar el embedding de la imagen enviada por el cliente en bytes.
def get_image_embedding_for_inference(image_bytes):
    #Abre la imagen y la convierte a RGB.
    print("Procesando imagen...")
    image = Image.open(image_bytes).convert("RGB")
    #image = image_bytes.convert("RGB")
    print("Imagen procesada.")
    #Procesamiento de la imagen.
    inputs = processor(images=image, return_tensors="pt").to(device)
    #Se obtiene el embedding.
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten().reshape(1, -1) # Devolver como numpy array 2D para cosine_similarity

# Función para identificar el Pokémon más similar a la imagen recibida.
def identify_pokemon(uploaded_image_bytes):
    if not POKEMON_DATASET:
        return "Error: Dataset de Pokémon no cargado. No se puede identificar."

    #Obtiene el embedding de la imagen enviada por el cliente.
    try:
        input_embedding = get_image_embedding_for_inference(uploaded_image_bytes)
    except Exception as e:
        return f"Error al procesar la imagen de entrada: {e}"

    best_match_pokemon = "Unknown"
    max_similarity = -1.0

    # Calcula la similitud de coseno para cada Pokémon en el dataset y obtiene el mejor match.
    for pokemon_name, ref_embedding in POKEMON_DATASET.items():
        # Se asegura que el embedding de referencia también sea 2D.
        ref_embedding_2d = ref_embedding.reshape(1, -1)
        
        # Calcula similitud de coseno.
        #similarity = torch.nn.functional.cosine_similarity(input_embedding, ref_embedding_2d).item()
        similarity = cosine_similarity(input_embedding, ref_embedding_2d)[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_pokemon = pokemon_name
    
    #Imprime el Pokémon más similar y su similitud.
    if max_similarity < 0.65: #Umbral de confianza
        print(f"No se pudo identificar el Pokémon (similitud máxima: {max_similarity:.2f}).")
        best_match_pokemon = "Unknown"
        return best_match_pokemon
    else:
        print(f"Pokémon detectado: {best_match_pokemon} (similitud: {max_similarity:.2f})")
        return best_match_pokemon

# Ruta principal que devuelve un mensaje de bienvenida.
@app.route('/')
def home():
    return jsonify("Hola Mundo!!!"), 200

# Ruta para identificar Pokémon a partir de una imagen enviada en formato Base64.
@app.route('/identify', methods=['POST'])
def identify():
    #Verfica que la petición sea JSON.
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    #Obtiene los datos JSON del cuerpo de la petición.
    data = request.get_json()
    #Verifica que el campo 'image' esté presente en los datos JSON.
    if 'image' not in data:
        return jsonify({"error": "No 'image' field in JSON data"}), 400
    
    imageDataUrl = data['image']

    #Extrae la parte Base64 y la decodifica.
    #El formato es típicamente "data:image/<tipo>;base64,<data_base64>"
    #Necesitamos quitar el prefijo "data:image/<tipo>;base64,"
    if "," in imageDataUrl:
        header, base64_encoded_image = imageDataUrl.split(",", 1)
    else:
        # Si no hay coma, asumimos que es solo la cadena base64 directamente
        base64_encoded_image = imageDataUrl

    try:
        # Decodificar la cadena Base64 a bytes binarios
        image_bytes = base64.b64decode(base64_encoded_image)
    except base64.binascii.Error as e:
        return jsonify({"error": f"Invalid Base64 string: {e}"}), 400
    
    #Creaa un objeto BytesIO con los bytes decodificados.
    image_bytes_io = io.BytesIO(image_bytes)
        
    # Llama a la función de detección.
    pokemon_name = identify_pokemon(image_bytes_io)
        
    # Puedes parsear el resultado si quieres devolver solo el nombre o la similitud por separado
    # Por ejemplo, si identify_pokemon devuelve "Pikachu (similitud: 0.92)"
    return jsonify({'message': pokemon_name}), 200

if __name__ == '__main__':
    app.run(debug = True)
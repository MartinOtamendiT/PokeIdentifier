import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import io

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

# Ejemplo de uso.
if __name__ == "__main__":
    
    test_image_path = "pokemon_dataset/pikachu/pikachu_3.jpg"
    if not os.path.exists(test_image_path):
        print(f"Error: La imagen de prueba '{test_image_path}' no existe. Por favor, crea una para probar.")
    else:
        with open(test_image_path, "rb") as f:
            image_bytes_from_upload = f.read()
        
        print(f"\nIdentificando Pokémon en '{test_image_path}'...")
        detected_pokemon = identify_pokemon(io.BytesIO(image_bytes_from_upload))
        print(f"Pokémon detectado: {detected_pokemon}")
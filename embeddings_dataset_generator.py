import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np

# Carga del modelo y del procesador CLIP.
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Se asegura de usar GPU si está disponible.
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Función para generar el embedding de una imagen usando CLIP.
def get_image_embedding(image_path):
    #Abre la imagen y la convierte a RGB.
    image = Image.open(image_path).convert("RGB")
    #Procesamiento de la imagen.
    inputs = processor(images=image, return_tensors="pt").to(device)
    #Se obtiene el embedding.
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten() # Devolver como numpy array plano

# Función encargada de crear un dataset de embeddings de las imágenes de los Pokémon.
def create_pokemon_embeddings_dataset(dataset_path="pokemon_dataset"):
    pokemon_embeddings = {}
    #Recorre cada carpeta de Pokémon en el dataset.
    for pokemon_name in os.listdir(dataset_path):
        #Concatena la ruta del directorio del Pokémon.
        pokemon_dir = os.path.join(dataset_path, pokemon_name)
        #Verifica si es el directorio existe.
        if os.path.isdir(pokemon_dir):
            print(f"Procesando {pokemon_name}...")
            avg_embedding = []
            #Recorre cada archivo listado dentro del directorio del Pokémon.
            for image_file in os.listdir(pokemon_dir):
                #Verifica si el archivo es una imagen.
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(pokemon_dir, image_file)
                    try:
                        #Genera el embedding de la imagen.
                        embedding = get_image_embedding(image_path)
                        avg_embedding.append(embedding)
                    except Exception as e:
                        print(f"Error procesando {image_path}: {e}")
            
            #Calcula el promedio de los embeddings si es que se encontraron imágenes válidas.
            if avg_embedding: 
                pokemon_embeddings[pokemon_name] = np.mean(avg_embedding, axis=0)
            else:
                print(f"No se encontraron imágenes válidas para {pokemon_name}")
    return pokemon_embeddings

if __name__ == "__main__":    
    print("Creando el dataset de embeddings de Pokémon...")
    pokemon_embeddings_dataset = create_pokemon_embeddings_dataset()

    #Guarda el dataset de embeddings para no tener que regenerarla cada vez
    np.save("pokemon_embeddings_dataset.npy", pokemon_embeddings_dataset)
    print("Dataset de embeddings de Pokémon creada y guardada en 'pokemon_embeddings_dataset.npy'")

    # Para verificar, puedes cargarla:
    # loaded_db = np.load("pokemon_embeddings.npy", allow_pickle=True).item()
    # print(f"Ejemplo de un embedding cargado (Pikachu si existe): {loaded_db.get('pikachu', 'No encontrado')[:5]}...") # Mostrar los primeros 5 elementos
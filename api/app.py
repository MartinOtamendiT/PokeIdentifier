import os
import json
from flask import Flask, request, render_template, session, jsonify, redirect, url_for,current_app, g, flash
from flask_cors import CORS
import pickle
from pokemon_identifier import identify_pokemon, POKEMON_DATASET
import io
import base64

#Configuración de la aplicación Flask
app = Flask(__name__)
CORS(app) #Esto habilita CORS para todas las rutas y orígenes.

@app.route('/')
def home():
    return jsonify("Hola Mundo!!!"), 200

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
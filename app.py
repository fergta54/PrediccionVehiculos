import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, jsonify, request, render_template
from PIL import Image
import io

app = Flask(__name__)

# Cargar los modelos
model_categoria = load_model('ModeloAutos.keras')
model_anio = load_model('ModeloAnio.keras')
model_marca = load_model('ModeloMarca.keras')

# Categorías para cada modelo
data_cat = ['bus', 'camion', 'camioneta', 'deportivo', 'hatchback', 'minibus', 'sedan', 'suv-larga', 'suv-ligera', 'todoterreno']
anio_cat = ['1960', '1970', '1980', '1990', '2000', '2010', '2020']
marca_cat = ['Audi', 'BMW', 'Chery', 'Chevrolet', 'Ford', 'Great Wall', 'JAC Motors', 'Kia', 'Mazda', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Peugeot', 'Renault', 'Subaru', 'Suzuki', 'Toyota', 'Volkswagen']
img_height = 180
img_width = 180

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).resize((img_width, img_height))
        img_arr = tf.keras.preprocessing.image.img_to_array(image)
        img_bat = np.expand_dims(img_arr, axis=0)

        # Realizar las predicciones
        predict_cat = model_categoria.predict(img_bat)
        predict_anio = model_anio.predict(img_bat)
        predict_marca = model_marca.predict(img_bat)

        score_cat = tf.nn.softmax(predict_cat[0])
        score_anio = tf.nn.softmax(predict_anio[0])
        score_marca = tf.nn.softmax(predict_marca[0])

        category = data_cat[np.argmax(score_cat)]
        anio = anio_cat[np.argmax(score_anio)]
        marca = marca_cat[np.argmax(score_marca)]

        confidence_cat = np.max(score_cat) * 100
        confidence_anio = np.max(score_anio) * 100
        confidence_marca = np.max(score_marca) * 100

        return jsonify({
            "category": category,
            "confidence_cat": confidence_cat,
            "año": anio,
            "confidence_anio": confidence_anio,
            "marca": marca,
            "confidence_marca": confidence_marca
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

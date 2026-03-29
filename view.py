import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
# Load the pre-trained MobileNet model
MODEL_PATH = 'stroke_detection/static/models/mobilenettt.h5'
model = load_model(MODEL_PATH)

def preprocess_image(image_path):
    """Preprocess the input image for prediction."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_stroke(request):
    """Handle image uploads and predict stroke."""
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(file_path)

        # Preprocess the uploaded image
        processed_image = preprocess_image(fs.path(file_path))
        prediction = model.predict(processed_image)

        # Interpret results
        result = "Stroke Detected" if prediction[0][0] > 0.5 else "No Stroke Detected"
        accuracy = f"Confidence: {round(prediction[0][0] * 100, 2)}%"

        return render(request, 'stroke_detection/result.html', {
            'file_url': file_url,
            'result': result,
            'accuracy': accuracy
        })

    return render(request, 'stroke_detection/upload.html')

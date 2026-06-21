from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load trained model
model = load_model("food_recognition_model.h5")

# Food classes
class_names = [
    "burger",
    "butter_naan",
    "chai",
    "chapati",
    "chole_bhature",
    "dal_makhani",
    "dhokla",
    "fried_rice",
    "idli",
    "jalebi",
    "kaathi_rolls",
    "kadai_paneer",
    "kulfi",
    "masala_dosa",
    "momos",
    "paani_puri",
    "pakode",
    "pav_bhaji",
    "pizza",
    "samosa"
]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    predict_image(image_path)

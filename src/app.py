import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TensorFlow Lite Model
interpreter = tf.lite.Interpreter(
    model_path="food_recognition_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

def predict(img):
    try:
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(
            input_details[0]['index'],
            img_array
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_details[0]['index']
        )[0]

        result = {
            class_names[i]: float(predictions[i])
            for i in range(len(class_names))
        }

        return result

    except Exception as e:
        return {"Error": str(e)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Food Image"),
    outputs=gr.Label(num_top_classes=3),
    title="Food Recognition System",
    description="Upload a food image and get predictions."
)

if __name__ == "__main__":
    demo.launch()

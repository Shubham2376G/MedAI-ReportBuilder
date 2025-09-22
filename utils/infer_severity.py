from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from tensorflow.keras.models import Model
import cv2

def severity_model(input_img_path,input_text,output_path):

    loaded_model = load_model("models/severity/multimodal_model.keras")
    print("Multimodal model loaded successfully.")

    # Define your input image path and text description
    input_image_path = input_img_path  # Replace with the path to your image
    input_text_description = input_text # Replace with your text
    image_size = (224, 224)
    # Preprocess the image
    img = Image.open(input_image_path).convert('RGB')
    img_array = np.array(img)
    img_array = tf.image.resize(img_array, image_size)
    img_array = tf.cast(img_array, tf.float32)
    preprocessed_image_input = tf.expand_dims(preprocess_input(img_array), axis=0)

    # Preprocess the text


    with open("models/severity/tokenizer.json", 'r') as f:
        tokenizer_json = f.read()  # Read as string

    loaded_tokenizer = tokenizer_from_json(tokenizer_json)


    input_text_sequence = loaded_tokenizer.texts_to_sequences([input_text_description])
    padded_input_text_sequence = pad_sequences(input_text_sequence, maxlen=100, padding='post', truncating='post')
    preprocessed_text_input = tf.expand_dims(padded_input_text_sequence[0], axis=0)


    # Make a prediction
    predictions = loaded_model.predict([preprocessed_image_input, preprocessed_text_input])

    # Get the predicted class label
    predicted_class_index = np.argmax(predictions[0])
    # Reverse the label map to get the class name
    label_map = {"NO DR": 0, "Mild DR": 1, "Moderate DR": 2, "Severe DR": 3, "Proliferative DR": 4} # Define your label mapping
    reverse_label_map = {v: k for k, v in label_map.items()}
    predicted_label = reverse_label_map[predicted_class_index]


    print(f"Input Image Path: {input_image_path}")
    print(f"Input Text Description: '{input_text_description}'")
    print(f"Predicted DR Severity: {predicted_label}")



    # Identify the last convolutional layer in the image encoder
    last_conv_layer = None
    # Iterate through the layers of the multimodal model in reverse order
    for layer in reversed(loaded_model.layers):
        # Check if the layer is a Conv2D layer and part of the image processing branch
        # We can identify image processing layers by checking if their inputs originate from 'image_input'
        # A simpler approach given the VGG16 structure is to find the last Conv2D layer before the text branch layers start
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break # Found the last Conv2D layer

    if last_conv_layer:
        print(f"Identified last convolutional layer: {last_conv_layer.name}")
    else:
        print("No Conv2D layer found in the image encoder.")



    # Define a new model for Grad-CAM visualization
    grad_cam_model = Model(
        inputs=loaded_model.inputs,
        outputs=[last_conv_layer.output, loaded_model.output]
    )

    print("Grad-CAM model built successfully.")


    # Calculate gradients using tf.GradientTape
    with tf.GradientTape() as tape:
        # Get feature maps and predictions
        last_conv_layer_output, predictions = grad_cam_model(
            (preprocessed_image_input, preprocessed_text_input)
        )

        # Get the predicted class index
        predicted_class_index = tf.argmax(predictions[0])

        # Get the score for the predicted class
        predicted_class_score = predictions[:, predicted_class_index]

    # Compute the gradient of the predicted class score with respect to the feature maps
    gradients = tape.gradient(predicted_class_score, last_conv_layer_output)

    print(f"Calculated gradients with shape: {gradients.shape}")
    print(f"Predicted class index: {predicted_class_index.numpy()}")

    # Compute the weighted average of the gradients and feature maps
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to the heatmap
    heatmap = tf.maximum(heatmap, 0)

    # Normalize the heatmap
    max_heatmap = tf.reduce_max(heatmap)
    if max_heatmap == 0:
        max_heatmap = 1e-10 # Avoid division by zero
    heatmap /= max_heatmap

    print(f"Calculated heatmap with shape: {heatmap.shape}")


    # Load the original image again to overlay the heatmap
    original_img = Image.open(input_img_path).convert("RGB") # Ensure image is in RGB
    original_img = np.array(original_img)

    # Resize the heatmap to the original image size
    heatmap = np.uint8(255 * heatmap)
    resized_heatmap = tf.image.resize(heatmap[..., tf.newaxis], (original_img.shape[0], original_img.shape[1]))
    resized_heatmap = tf.squeeze(resized_heatmap).numpy()

    # Ensure resized_heatmap is of type uint8 for cv2 operations
    resized_heatmap = np.uint8(resized_heatmap)

    # Apply a color map to the heatmap (e.g., JET)
    heatmap_img = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(original_img.astype(np.uint8), 0.6, heatmap_img.astype(np.uint8), 0.4, 0)

    print("Heatmap resized and ready for visualization.")

    cv2.imwrite(output_path, superimposed_img)


    return predicted_label



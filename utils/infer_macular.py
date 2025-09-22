import tensorflow as tf
import cv2
import numpy as np
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def macular_model(image_path, output_grad):
    # Load the saved model
    loaded_model = tf.keras.models.load_model('models/macular/eye_fundus_cnn_model.keras')
    print(loaded_model.summary())
    print("Model loaded successfully.")

    # Define a function to preprocess a single image for inference
    def preprocess_image_for_inference(image_path, img_height, img_width):
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0) # Add batch dimension
            return img
        else:
            print(f"Error: Could not load image {image_path}")
            return None



    # Preprocess the sample image
    preprocessed_image = preprocess_image_for_inference(image_path, 128,128)

    if preprocessed_image is not None:
        # Make a prediction
        prediction = loaded_model.predict(preprocessed_image)

        # Interpret the prediction (assuming binary classification with sigmoid output)
        predicted_class = (prediction > 0.5).astype(int)

        print(f"Raw prediction output: {prediction}")
        print(f"Predicted class (0 or 1): {predicted_class[0][0]}")

        # Set last convolutional layer
        last_conv_layer_name = "conv2d_2"

        # Load and preprocess image
        def load_preprocess_image(img_path, target_size=(128,128)):
            img = tf.keras.utils.load_img(img_path, target_size=target_size)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            return img_array, img

        # Grad-CAM function
        def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
            # Create model that maps input to activations of the last conv layer
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=model.get_layer(last_conv_layer_name).output
            )

            with tf.GradientTape() as tape:
                # Forward pass
                conv_outputs = grad_model(img_array)
                tape.watch(conv_outputs)

                # Manually forward through the remaining layers to get the scalar output
                x = conv_outputs
                for layer in model.layers[model.layers.index(model.get_layer(last_conv_layer_name)) + 1:]:
                    x = layer(x)

                class_channel = x[:, 0]  # if it's a single sigmoid neuron

            # Compute gradients
            grads = tape.gradient(class_channel, conv_outputs)

            # Pool gradients across spatial dimensions
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            # Weight the channels
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + tf.keras.backend.epsilon())

            return heatmap.numpy()

        # Superimpose heatmap on original image
        def display_gradcam(img_path, heatmap, alpha=0.4):
            _, original_img = load_preprocess_image(img_path)
            img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)

            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)

            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

            cv2.imwrite(output_grad, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))


        # Run on an image
        img_path = image_path  # Replace this
        img_array, _ = load_preprocess_image(img_path)
        heatmap = make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name)
        display_gradcam(img_path, heatmap)





        return predicted_class[0][0]






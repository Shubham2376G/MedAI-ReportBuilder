import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use only GPU 3

import tensorflow as tf

# If using SSIM loss, you need to re-register it
from tensorflow.image import ssim

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))


# Combined loss (SSIM + MSE)
def combined_loss(y_true, y_pred):
    ssim_component = ssim_loss(y_true, y_pred)
    mse_component = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.7 * ssim_component + 0.3 * mse_component

# Load model (with custom_objects if needed)
model = tf.keras.models.load_model("models/artifact_remove/improved_arts.keras", custom_objects={'ssim_loss': ssim_loss, 'combined_loss': combined_loss})

import cv2
import numpy as np

def load_and_preprocess(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def fuse_images(img_path1, img_path2, model, target_size=(256, 256)):
    img1 = load_and_preprocess(img_path1, target_size)
    img2 = load_and_preprocess(img_path2, target_size)
    
    input_concat = np.concatenate([img1, img2], axis=-1)
    input_batch = np.expand_dims(input_concat, axis=0)

    pred = model.predict(input_batch)[0]
    pred = np.clip(pred, 0, 1)

    pred_uint8 = (pred * 255).astype(np.uint8)
    return pred_uint8


output = fuse_images( "l.jpg", "r.jpg",model)
cv2.imwrite("infered_imageOD_2.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


# output2 = fuse_images( "test/volk_2img_inference_RESIZED/T_Keerthika_keerthika2306/20250623_165537_OD.jpg", "test/volk_2img_inference_RESIZED/T_Keerthika_keerthika2306/20250623_165451_OD.jpg",model)
# cv2.imwrite("infered_imageOD_2.jpg", cv2.cvtColor(output2, cv2.COLOR_RGB2BGR))


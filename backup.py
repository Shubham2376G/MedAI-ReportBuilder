from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt




# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "models/outs", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained("models/outs")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "data/uploads/patient_5/right_l.jpg",
            },
            {"type": "text", "text": "This is a macula-centered fundus image of the eye. Clinical observations include: Copper Wiring, Tessellated Fundus/Tigroid Fundus.\n\nDetermine the diabetic retinopathy grade (No DR, Mild, Moderate, Severe, or Proliferative)."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=10)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)



model.eval() # Set the model to evaluation mode


target_layer = model.transformer.h[-1].mlp.fc2 # Example target layer, replace with your chosen layer
target_class_index = 0 # Replace with the index of your target class

# Step 4: Forward pass with gradient tracking
inputs["pixel_values"].requires_grad = True
outputs = model(**inputs)

# Step 5: Calculate gradients
# Assuming your model outputs logits for classification
logits = outputs.logits
target_class_logit = logits[0, target_class_index]
target_class_logit.backward()

# Step 6: Generate Grad-CAM heatmap
gradients = target_layer.activations.grad
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
activations = target_layer.activations
for i in range(activations.size(1)):
    activations[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Step 7: Visualize Grad-CAM
image = cv2.imread(image_path)
heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

# Display the original image and the Grad-CAM visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM")
plt.axis("off")

plt.show()
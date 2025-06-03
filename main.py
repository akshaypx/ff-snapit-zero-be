import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load model and processor
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# Load your image
image_path = "image.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Text queries: objects you want to detect
texts = ["sofa", "lamp", "painting", "potted plant", "table", "rug", "blanket", "pillow"]  # Add your objects here

# Preprocess
inputs = processor(text=texts, images=image, return_tensors="pt")

# Run model
with torch.no_grad():
    outputs = model(**inputs)

# Get results
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)[0]

# Visualize
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
    ax.add_patch(patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none'))
    ax.text(x, y, f"{texts[label]}: {round(score.item(), 2)}", color='white', backgroundcolor='red', fontsize=10)

plt.axis("off")
plt.show()

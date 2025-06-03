from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

import random
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torchvision.transforms as T

app = FastAPI()

# Allow CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OWL-ViT model and processor once
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# Define object queries
DETECTION_TEXTS = ["sofa", "lamp", "painting", "potted plant", "table", "rug", "blanket", "pillow", "coffee table", "center table", "armchair", "chair"]

# Response schema
class DetectionResponse(BaseModel):
    annotated_image: str
    detected_items: List[str]

def read_image(file: UploadFile) -> Image.Image:
    return Image.open(BytesIO(file.file.read())).convert("RGB")

def encode_image(img_array: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode(".jpg", img_rgb)
    if not success:
        raise ValueError("Image encoding failed")
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

# Pre-generate distinct colors for each class
def get_unique_colors(num_colors, seed=42):
    random.seed(seed)
    return {
        i: tuple(random.choices(range(50, 256), k=3))  # Avoid very dark colors
        for i in range(num_colors)
    }

# Assign colors to your detection classes
CLASS_COLORS = get_unique_colors(len(DETECTION_TEXTS))

def annotate_image(image: Image.Image, results, texts):
    image_np = np.array(image).copy()
    height, width = image_np.shape[:2]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        x0, y0, x1, y1 = box

        # Dynamic styling
        box_thickness = max(2, int(min(width, height) * 0.003))
        font_scale = max(0.5, min(width, height) * 0.0015)

        # Get unique color for class
        color = CLASS_COLORS[label.item()]
        color_rgb = (int(color[0]), int(color[1]), int(color[2]))

        # Draw bounding box
        cv2.rectangle(image_np, (x0, y0), (x1, y1), color_rgb, box_thickness)

        # Draw label background
        label_text = f"{texts[label]} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(image_np, (x0, y0 - text_height - 6), (x0 + text_width, y0), color_rgb, -1)

        # Draw text
        cv2.putText(image_np, label_text, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    return image_np

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    image = read_image(file)
    inputs = processor(text=DETECTION_TEXTS, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

    detected_labels = [DETECTION_TEXTS[label] for label in results["labels"]]
    annotated_img = annotate_image(image, results, DETECTION_TEXTS)

    return {
        "annotated_image": encode_image(annotated_img),
        "detected_items": detected_labels,
    }

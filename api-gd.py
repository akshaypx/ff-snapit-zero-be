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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

app = FastAPI()

# Allow CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Grounding DINO model and processor once
model_id = "IDEA-Research/grounding-dino-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Define object queries
DETECTION_TEXTS = [
    "sofa", "lamp", "painting", "plant", "rug", "mirror"
,    "blanket", "cushion", "table", "chair", "curtain", "clock", "bed", "shelf"
]
# Grounding DINO expects queries to be lowercased and end with a period
DETECTION_TEXT = ". ".join([t.lower() for t in DETECTION_TEXTS]) + "."

# Response schema
class DetectionResponse(BaseModel):
    annotated_image: str
    detected_items: List[str]

def get_color_for_label(label: str) -> tuple:
    random.seed(hash(label) % 10000)  # Ensures same label always gets same color
    return tuple(random.choices(range(50, 256), k=3))

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

def annotate_image(image: Image.Image, results):
    image_np = np.array(image).copy()
    height, width = image_np.shape[:2]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        x0, y0, x1, y1 = box

        box_thickness = max(2, int(min(width, height) * 0.003))
        font_scale = max(0.5, min(width, height) * 0.0015)

        color = get_color_for_label(label)
        cv2.rectangle(image_np, (x0, y0), (x1, y1), color, box_thickness)

        label_text = f"{label} ({score:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(image_np, (x0, y0 - text_height - 6), (x0 + text_width, y0), color, -1)
        cv2.putText(image_np, label_text, (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    return image_np

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    image = read_image(file)
    inputs = processor(images=image, text=DETECTION_TEXT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=target_sizes
    )[0]

    detected_labels = results["labels"]
    annotated_img = annotate_image(image, results)

    return {
        "annotated_image": encode_image(annotated_img),
        "detected_items": detected_labels,
    }

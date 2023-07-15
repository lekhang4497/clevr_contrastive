from PIL import Image
import requests
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)
import torch
import json
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MODEL = "models/vit-b-p16-224-roberta-b"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
image_processor = AutoImageProcessor.from_pretrained(MODEL)
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
model = VisionTextDualEncoderModel.from_pretrained(MODEL)
model.to(device)

shapes = ["cube", "sphere", "cylinder"]
colors = ["gray", "red", "blue", "green", "yellow"]
materials = ["rubber", "metal"]

obj_classes = [
    f"{color} {material} {shape}"
    for shape in shapes
    for color in colors
    for material in materials
]

with open("data/test_unseen_shape.json") as f:
    test_data = [json.loads(line) for line in f]
img_dir = "clevr_images/unseen_shape"

for d in test_data:
    d["image_path"] = os.path.join(img_dir, d["image"])
    d["label"] = f"{d['color']} {d['material']} {d['shape']}"

images = [Image.open(d["image_path"]).convert("RGB") for d in test_data]
inputs = processor(
    text=obj_classes, images=images, return_tensors="pt", padding=True
).to(device)

with torch.no_grad():
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
    )
    logits_per_image = outputs.logits_per_image

t = torch.max(logits_per_image, dim=1)
predict = [obj_classes[i] for i in t.indices]

count = 0
for i, d in enumerate(test_data):
    if d["label"] == predict[i]:
        count += 1

print(f"{count}/{len(test_data)}")

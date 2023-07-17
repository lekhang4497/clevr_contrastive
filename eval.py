from typing import List
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from collections import defaultdict

logger = logging.getLogger("eval_logger")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("eval.log")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_formatter = logging.Formatter("%(levelname)s - %(message)s")

file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MODEL = "models/vit-b-p16-224-roberta-b"
EVAL_JSON = "data/test_unseen_shape.json"
EVAL_IMG_DIR = "clevr_images/unseen_shape"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
image_processor = AutoImageProcessor.from_pretrained(MODEL)
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
model = VisionTextDualEncoderModel.from_pretrained(MODEL)
model.to(device)

# shapes = ["cube", "sphere", "cylinder"]
shapes = ["diamond", "cone", "donut"]
colors = ["gray", "red", "blue", "green", "yellow"]
materials = ["rubber", "metal"]

obj_classes = [
    f"{color} {material} {shape}"
    for shape in shapes
    for color in colors
    for material in materials
]


def extract_properties(preds: List[str]):
    colors = []
    materials = []
    shapes = []
    for pred in preds:
        color, material, shape = pred.split()
        colors.append(color)
        materials.append(material)
        shapes.append(shape)
    return {"color": colors, "material": materials, "shape": shapes}


with open(EVAL_JSON) as f:
    test_data = [json.loads(line) for line in f]

for d in test_data:
    d["image_path"] = os.path.join(EVAL_IMG_DIR, d["image"])
    d["label"] = f"{d['color']} {d['material']} {d['shape']}"

data_loader = DataLoader(test_data, batch_size=1000)

correct = defaultdict(int)
for batch in tqdm(data_loader):
    images = [Image.open(img_path).convert("RGB") for img_path in batch["image_path"]]
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
        preds = [obj_classes[i] for i in t.indices]

        pred_props = extract_properties(preds)

        for prop in ["color", "material", "shape"]:
            correct[prop] += sum(
                [
                    1
                    for pred, label in zip(pred_props[prop], batch[prop])
                    if pred == label
                ]
            )
        correct["label"] += sum(
            [1 for pred, label in zip(preds, batch["label"]) if pred == label]
        )
        for pred, label in zip(preds, batch["label"]):
            logger.debug(f"Label: {label}\nPred: {pred}\n\n")

    # count = 0
    # for i, d in enumerate(test_data):
    #     if d["label"] == predict[i]:
    #         count += 1

# logger.info(f"{correct}/{len(test_data)}")
logger.info(json.dumps(correct, indent=4))
logger.info(json.dumps({k: v / len(test_data) for k, v in correct.items()}, indent=4))

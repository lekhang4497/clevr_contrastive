from typing import List
from PIL import Image
import requests
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    AutoProcessor,
    CLIPModel,
    OwlViTProcessor,
    OwlViTModel,
)
import torch
import json
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from collections import defaultdict
from datetime import datetime
import pandas as pd


logger = logging.getLogger("eval_logger")
logger.setLevel(logging.DEBUG)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"logs/eval_log_{current_time}.txt"
file_handler = logging.FileHandler(log_file)
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

# MODEL = "models/vit-b-p16-224-roberta-b-subset1000"
MODEL = "prop_clip_models/vit-b-p16-224-roberta-b-tr1000"
# MODEL = "prop_clip_models/propclip-init-vit-b-p16-224-roberta-b"
# MODEL = "prop_clip_models/propclip-init-specprop-tr1000"
# MODEL = "models/vit-b-p16-224-roberta-b-specprop-tr1000"
# MODEL = "openai/clip-vit-base-patch32"
SETTING = "unseen_shape"

TOP_K = 500

if SETTING == "normal":
    EVAL_JSON = "data/test.json"
    EVAL_IMG_DIR = "clevr_images/normal"
    shapes = ["cube", "sphere", "cylinder"]
    # shapes = ["diamond", "cone", "donut"]
    colors = ["gray", "red", "blue", "green", "yellow"]
    # colors = ["brown", "purple", "cyan"]
    materials = ["rubber", "metal"]
elif SETTING == "unseen_shape":
    EVAL_JSON = "data/test_unseen_shape.json"
    EVAL_IMG_DIR = "clevr_images/unseen_shape"
    shapes = ["diamond", "cone", "donut"]
    colors = ["gray", "red", "blue", "green", "yellow"]
    materials = ["rubber", "metal"]
elif SETTING == "unseen_color":
    EVAL_JSON = "data/test_unseen_color.json"
    EVAL_IMG_DIR = "clevr_images/unseen_color"
    shapes = ["cube", "sphere", "cylinder"]
    colors = ["brown", "purple", "cyan"]
    materials = ["rubber", "metal"]

CACHE_DIR = "cache"

if "openai/clip" in MODEL:
    model = CLIPModel.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    processor = AutoProcessor.from_pretrained(MODEL, cache_dir=CACHE_DIR)
elif "owlvit" in MODEL:
    model = OwlViTModel.from_pretrained(
        "google/owlvit-base-patch32", cache_dir=CACHE_DIR
    )
    processor = OwlViTProcessor.from_pretrained(
        "google/owlvit-base-patch32", cache_dir=CACHE_DIR
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, cache_dir=CACHE_DIR)
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
    model = VisionTextDualEncoderModel.from_pretrained(MODEL, cache_dir=CACHE_DIR)
model.to(device)


with open(EVAL_JSON) as f:
    test_data = [json.loads(line) for line in f]

for d in test_data:
    d["image_path"] = os.path.join(EVAL_IMG_DIR, d["image"])
    d["label"] = f"{d['color']} {d['material']} {d['shape']}"


@torch.no_grad()
def get_image_embeddings(model, images: List[Image.Image]):
    inputs = processor(text=[""], images=images, return_tensors="pt", padding=True).to(
        device
    )
    outputs = model(**inputs)
    return outputs.image_embeds


@torch.no_grad()
def get_text_embeddings(model, texts: List[str]):
    inputs = processor(
        text=texts, images=[Image.new("RGB", (2, 2))], return_tensors="pt", padding=True
    ).to(device)
    outputs = model(**inputs)
    return outputs.text_embeds


def retrieve_img(image_embeddings, text, top_k=TOP_K):
    text_embedding = get_text_embeddings(model, [text])
    logits_per_text = torch.matmul(text_embedding, image_embeddings.t())
    return torch.topk(logits_per_text[0], k=top_k).indices


data_loader = DataLoader(test_data, batch_size=1000)

image_embeddings = None
for batch in tqdm(data_loader):
    images = [Image.open(img_path).convert("RGB") for img_path in batch["image_path"]]
    batch_image_embeddings = get_image_embeddings(model, images)
    if image_embeddings is None:
        image_embeddings = batch_image_embeddings
    else:
        image_embeddings = torch.cat((image_embeddings, batch_image_embeddings), dim=0)


# query = "The color is green. The material is metal"
# # query = "A photo of a green metal object"
# retrieved_indices = retrieve_img(image_embeddings, query)

# correct = 0
# for i in retrieved_indices:
#     if test_data[i]["material"] == "metal" and test_data[i]["color"] == "green":
#         correct += 1

# print(f"Correct: {correct}/{len(retrieved_indices)}")

print(f"Eval for model: {MODEL}, top-k={TOP_K}")
df = pd.DataFrame(index=[x for x in colors], columns=[x for x in materials])
for color in colors:
    for material in materials:
        query = f"The color is {color}. The material is {material}"
        # query = f"A photo of a {color} {material} object"
        retrieved_indices = retrieve_img(image_embeddings, query)
        correct = sum(
            1
            for i in retrieved_indices
            if test_data[i]["material"] == material and test_data[i]["color"] == color
        )
        relevant = sum(
            1 for d in test_data if d["material"] == material and d["color"] == color
        )
        precision = correct / len(retrieved_indices)
        recall = correct / relevant
        f1 = 2 * (recall * precision) / (recall + precision)
        print(f"Correct {correct}, relevant {relevant}")
        # df[material][color] = f"{correct}/{len(retrieved_indices)}"
        df[material][color] = f1

print(df)

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

# MODEL = "models/vit-b-xlmr-b-subset100"
MODEL = "models/vit-b-p16-224-roberta-b-subset1000"
# MODEL = "google/owlvit-base-patch32"
SETTING = "unseen_shape"

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


en_vi_dict = {
    "red": "màu đỏ",
    "gray": "màu xám",
    "blue": "màu xanh dương",
    "green": "màu xanh lá",
    "yellow": "màu vàng",
    "cube": "khối vuông",
    "sphere": "khối cầu",
    "cylinder": "khối trụ",
    "rubber": "cao su",
    "metal": "kim loại",
}

vi_en_dict = {v: k for k, v in en_vi_dict.items()}

if "clip" in MODEL:
    model = CLIPModel.from_pretrained(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL)
elif "owlvit" in MODEL:
    model = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL)
    processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
    model = VisionTextDualEncoderModel.from_pretrained(MODEL)
model.to(device)


def extract_properties(preds: List[str]):
    colors = []
    materials = []
    shapes = []
    for pred in preds:
        color, material, shape = pred.split()
        colors.append(color)
        materials.append(material)
        shapes.append(shape)
    return colors, materials, shapes


def extract_vi_properties(preds: List[str]):
    colors = []
    materials = []
    shapes = []
    for pred in preds:
        # Map back to english
        for k, v in vi_en_dict.items():
            pred = pred.replace(k, v)
        # Note that the order is shape -> material -> color
        shape, material, color = pred.split()
        colors.append(color)
        materials.append(material)
        shapes.append(shape)
    return colors, materials, shapes


with open(EVAL_JSON) as f:
    test_data = [json.loads(line) for line in f]

for d in test_data:
    d["image_path"] = os.path.join(EVAL_IMG_DIR, d["image"])
    d["label"] = f"{d['color']} {d['material']} {d['shape']}"


@torch.no_grad()
def classify_text_best_match_vision(model, images: List[Image.Image], texts: List[str]):
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(
        device
    )

    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values,
    )
    logits_per_image = outputs.logits_per_image

    max_indices = torch.max(logits_per_image, dim=1).indices.tolist()
    preds = [texts[i] for i in max_indices]
    assert len(preds) == len(images)
    return preds, max_indices


def get_vi_caption(color, material, shape):
    # color, material, shape = caption.split()
    color = en_vi_dict[color]
    material = en_vi_dict[material]
    shape = en_vi_dict[shape]
    return f"{shape} {material} {color}"


# def vi_label_caption_to_en(caption):
#     for k, v in vi_en_dict.items():
#         caption = caption.replace(k, v)
#     shape, material, color = caption.split()
#     return f"{color} {material} {shape}"


def eval_classify_by_property_label(test_data, map_language=None):
    """Eval by classying property label. E.g. classify as "red metal cube", "blue rubber sphere"

    Returns:
        Dict[str, int]: key is property, value is the number of correct classification
    """

    data_loader = DataLoader(test_data, batch_size=1000)

    if map_language == "vi":
        obj_classes = [
            get_vi_caption(color, material, shape)
            for shape in shapes
            for color in colors
            for material in materials
        ]
    else:
        obj_classes = [
            f"{color} {material} {shape}"
            for shape in shapes
            for color in colors
            for material in materials
        ]
    logger.info(f"Object classes: {obj_classes}")

    correct = defaultdict(int)

    for batch in tqdm(data_loader):
        images = [
            Image.open(img_path).convert("RGB") for img_path in batch["image_path"]
        ]
        # inputs = processor(
        #     text=obj_classes, images=images, return_tensors="pt", padding=True
        # ).to(device)

        # outputs = model(
        #     input_ids=inputs.input_ids,
        #     attention_mask=inputs.attention_mask,
        #     pixel_values=inputs.pixel_values,
        # )
        # logits_per_image = outputs.logits_per_image

        # t = torch.max(logits_per_image, dim=1)
        # preds = [obj_classes[i] for i in t.indices]
        preds, _ = classify_text_best_match_vision(model, images, obj_classes)
        if map_language == "vi":
            pred_colors, pred_mat, pred_shapes = extract_vi_properties(preds)
        else:
            pred_colors, pred_mat, pred_shapes = extract_properties(preds)

        # for prop in ["color", "material", "shape"]:
        #     correct[prop] += sum(
        #         [
        #             1
        #             for pred, label in zip(pred_props[prop], batch[prop])
        #             if pred == label
        #         ]
        #     )
        correct["color"] += sum(
            [1 for pred, label in zip(pred_colors, batch["color"]) if pred == label]
        )
        correct["material"] += sum(
            [1 for pred, label in zip(pred_mat, batch["material"]) if pred == label]
        )
        correct["shape"] += sum(
            [1 for pred, label in zip(pred_shapes, batch["shape"]) if pred == label]
        )
        correct["all"] += sum(
            [1 for pred, label in zip(preds, batch["label"]) if pred == label]
        )
        for pred, label in zip(preds, batch["label"]):
            logger.debug(f"Label: {label}\nPred: {pred}\n\n")
    return correct
    # count = 0
    # for i, d in enumerate(test_data):
    #     if d["label"] == predict[i]:
    #         count += 1


def eval_classify_by_separate_property(test_data):
    """Eval by classying separate property. E.g. classify "the material is metal", "the color is red", "the shape is a cube"

    Returns:
        Dict[str, int]: key is property, value is the number of correct classification
    """

    data_loader = DataLoader(test_data, batch_size=1000)

    material_classes = [f"The material is {material}" for material in materials]
    color_classes = [f"The color is {color}" for color in colors]
    shape_classes = [f"The shape is a {shape}" for shape in shapes]
    logger.info(f"Material classes: {material_classes}")
    logger.info(f"Color classes: {color_classes}")
    logger.info(f"Shape classes: {shape_classes}")

    correct = defaultdict(int)

    for batch in tqdm(data_loader):
        images = [
            Image.open(img_path).convert("RGB") for img_path in batch["image_path"]
        ]
        # Classify color
        _, indices = classify_text_best_match_vision(model, images, color_classes)
        pred_colors = [colors[i] for i in indices]
        # Classify material
        _, indices = classify_text_best_match_vision(model, images, material_classes)
        pred_mat = [materials[i] for i in indices]
        # Classify shape
        _, indices = classify_text_best_match_vision(model, images, shape_classes)
        pred_shapes = [shapes[i] for i in indices]

        preds = [
            f"{color} {mat} {shape}"
            for color, mat, shape in zip(pred_colors, pred_mat, pred_shapes)
        ]

        correct["color"] += sum(
            [1 for pred, label in zip(pred_colors, batch["color"]) if pred == label]
        )
        correct["material"] += sum(
            [1 for pred, label in zip(pred_mat, batch["material"]) if pred == label]
        )
        correct["shape"] += sum(
            [1 for pred, label in zip(pred_shapes, batch["shape"]) if pred == label]
        )
        correct["all"] += sum(
            [1 for pred, label in zip(preds, batch["label"]) if pred == label]
        )
        for pred, label in zip(preds, batch["label"]):
            logger.debug(f"Label: {label}\nPred: {pred}\n\n")
    return correct


# correct = eval_classify_by_property_label(test_data)
correct = eval_classify_by_separate_property(test_data)

# logger.info(f"{correct}/{len(test_data)}")
logger.info(json.dumps(correct, indent=4))
logger.info(json.dumps({k: v / len(test_data) for k, v in correct.items()}, indent=4))

import json
from PIL import Image
import os
from tqdm import tqdm
import glob
from datetime import datetime

CLEVR_GEN_DIR = (
    "/home/khangln/DRIVE_H/WORK/clevr_math/clevr-math/output_unseen_shape_5000"
)
OUT_IMG_DIR = "clevr_images/unseen_shape"

os.makedirs(OUT_IMG_DIR, exist_ok=True)


def process_raw_image(
    img_name, output_idx, output_img_dir=OUT_IMG_DIR, output_img_prefix="img_"
):
    IMG_PATH = f"{CLEVR_GEN_DIR}/images/{img_name}.png"
    SCENE_PATH = f"{CLEVR_GEN_DIR}/scenes/{img_name}.json"

    with open(SCENE_PATH) as f:
        scene = json.load(f)

    image = Image.open(IMG_PATH)

    scene_obj = scene["objects"][0]

    obj_info = {key: scene_obj[key] for key in ["color", "material", "shape"]}

    bounding_box = [
        scene_obj["x"],
        scene_obj["y"],
        scene_obj["width"],
        scene_obj["height"],
    ]

    bounding_box[2] += bounding_box[0]
    bounding_box[3] += bounding_box[1]

    # Crop the image
    cropped_image = image.crop(bounding_box)
    output_img_name = f"{output_img_prefix}{output_idx}.png"
    output_img_path = os.path.join(output_img_dir, output_img_name)
    cropped_image.save(output_img_path, "PNG")
    obj_info["image"] = output_img_name
    return obj_info


infos = []
num_gen = len(glob.glob(os.path.join(CLEVR_GEN_DIR, "images", "*.png")))
print(f"Processing {num_gen} files")
for i in tqdm(range(num_gen)):
    img_name = f"CLEVR_new_{str(i).zfill(6)}"
    infos.append(process_raw_image(img_name, output_idx=i))

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

out_file = f"obj_info_{formatted_datetime}.json"
with open(out_file, "w") as f:
    f.write("\n".join(json.dumps(x) for x in infos))

# Example Inference of GeoPixel on image data. Set query and image input and the model will return predictions with response description.
# The model require at least two 24 VRAM GPUs and takes few minutes to download, initialize and around 30sec inference.
# SpaceKnow Inc.

import os
from PIL import Image
import torch
import transformers
from torchvision import transforms
import re
import torch
import random
import numpy as np

from model.geopixel import GeoPixelForCausalLM


def rgb_color_text(text, r, g, b):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


# setup Model
version = "MBZUAI/GeoPixel-7B"
print(f"initialing tokenizer from: {version}")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    version,
    cache_dir=None,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.unk_token
seg_token_idx, bop_token_idx, eop_token_idx = [
    tokenizer(token, add_special_tokens=False).input_ids[0]
    for token in ["[SEG]", "<p>", "</p>"]
]

kwargs = {"torch_dtype": torch.bfloat16}
geo_model_args = {
    "vision_pretrained": "facebook/sam2-hiera-large",
    "seg_token_idx": seg_token_idx,  # segmentation token index
    "bop_token_idx": bop_token_idx,  # begining of phrase token index
    "eop_token_idx": eop_token_idx,  # end of phrase token index
}

# Load model
print(f"Load model from: {version}")

model = GeoPixelForCausalLM.from_pretrained(
    version, low_cpu_mem_usage=True, device_map="auto", **kwargs, **geo_model_args
)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.tokenizer = tokenizer


# Inference
query = f"find all planes on the image and output interleaved segmentation masks for the corresponding phrases."

image_path = "images/example1.png"
img = Image.open(image_path).resize((560, 560))

# normalize
visual_procesor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

img = visual_procesor(img).unsqueeze(0).unsqueeze(0).bfloat16()
device_type = next(iter(model.parameters())).device.type
img = img.to(device_type)

response, masks = model.evaluate(tokenizer, query, img, max_new_tokens=1024, hd_num=1)

pred_masks = masks
if pred_masks and "[SEG]" in response:
    # if pred_masks:
    pred_masks = pred_masks[0]
    pred_masks = pred_masks.detach().cpu().numpy()
    pred_masks = pred_masks > 0
    image_np = np.asarray(Image.open(image_path).resize((560, 560)))
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    save_img = image_np.copy()
    pattern = r"<p>(.*?)</p>\s*\[SEG\]"
    matched_text = re.findall(pattern, response)
    phrases = [text.strip() for text in matched_text]

    for i in range(pred_masks.shape[0]):
        mask = pred_masks[i]

        color = [random.randint(0, 255) for _ in range(3)]
        if matched_text:
            phrases[i] = rgb_color_text(phrases[i], color[0], color[1], color[2])
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        color_mask = np.array(color, dtype=np.uint8) * mask_rgb

        save_img = np.where(
            mask_rgb, (save_img * 0.5 + color_mask * 0.5).astype(np.uint8), save_img
        )
    if matched_text:
        split_desc = response.split("[SEG]")
        cleaned_segments = [
            re.sub(r"<p>(.*?)</p>", "", part).strip() for part in split_desc
        ]
        reconstructed_desc = ""
        for i, part in enumerate(cleaned_segments):
            reconstructed_desc += part + " "
            if i < len(phrases):
                reconstructed_desc += phrases[i] + " "
        print(reconstructed_desc)
    else:
        print(response.replace("\n", "").replace("  ", " "))

    save_path = "{}/{}_masked.jpg".format("vis_output", image_path.split("/")[-1])

    print("{} has been saved.".format(save_path))
else:
    print(response.replace("\n", "").replace("  ", " "))

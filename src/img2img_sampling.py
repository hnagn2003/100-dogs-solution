import requests
import torch
from PIL import Image, ImageDraw
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline
NUM_OF_LABEL = 1000
def create_mask(bbox_list, label_range=100):
    # Determine the image size based on bounding boxes
    max_x = max(bbox[2] for bbox in bbox_list)
    max_y = max(bbox[3] for bbox in bbox_list)

    # Create a blank RGB image with black background
    image = Image.new('RGB', (max_x, max_y), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes with normalized labels
    for bbox in bbox_list:
        bbox_top_left_x, bbox_top_left_y, bbox_bottom_right_x, bbox_bottom_right_y, label = bbox
        normalized_label = int((label / label_range) * 255)  # Normalize label to 0-255 range
        color = (normalized_label, normalized_label, normalized_label)
        draw.rectangle([bbox_top_left_x, bbox_top_left_y, bbox_bottom_right_x, bbox_bottom_right_y],
                       fill=color)

    return image
device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

mask = create_mask([(0, 40, 100, 80, 18), (350, 400, 510, 510, 18)])

prompt = "A fantasy landscape, have an astronaut riding a horse on mars"

# images = pipe(prompt=prompt, image=mask, strength=0.75, guidance_scale=7.5).images
mask.save("fantasy_landscape.png")
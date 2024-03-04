import torch
from diffusion_pipeline import StableDiffusionAnnotationPipeline
def main():
    IMG_SIZE = 512
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    pipe = StableDiffusionAnnotationPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to(device)
    prompt = "a photo of an astronaut riding a horse on mars"
    annotation = ["label_74_bbox_455.98_436.73_58.57_36.36","label_74_bbox_405.44_594.41_76.59_40.23"] #[tl_x, tl_y, br_x, br_y, label (18 is dog in COCO)]
    pipe.enable_attention_slicing()
    image = pipe(prompt=prompt, annotation=annotation).images[0]
    image.save("text2img_sample.jpg")
if __name__ == "__main__":
    main()
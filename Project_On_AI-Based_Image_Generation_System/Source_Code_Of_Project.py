import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

def generate_image(prompt):
    image = pipe(
        prompt,
        num_inference_steps=15, 
        height=512,              
        width=512
    ).images[0]
    return image

iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(type="pil"),
    title="AI Based Image Generator",
    description="Generate images from text prompts using a lightweight Stable Diffusion model"
)

iface.launch(share=True)
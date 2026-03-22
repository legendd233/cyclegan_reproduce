"""
CycleGAN Monet Style Transfer Demo

Transforms photos into Monet-style paintings using a trained CycleGAN model.
Built with Gradio for web deployment via Docker.
"""

import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
from model import load_generator

WEIGHT_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "checkpoints", "monet_cyclegan", "latest_net_G_B.pth"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {WEIGHT_PATH} (device: {DEVICE})")
generator = load_generator(WEIGHT_PATH, device=DEVICE)


def transform_to_monet(input_image: Image.Image) -> Image.Image:
    """Transform input photo to Monet style"""
    original_size = input_image.size  # (w, h)
    img = input_image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor - 0.5) / 0.5
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        output = generator(img_tensor)

    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    output = ((output + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    result = Image.fromarray(output)
    result = result.resize(original_size, Image.Resampling.BICUBIC)
    return result


monet_theme = gr.themes.Base(
    primary_hue=gr.themes.Color(c50="#fdf6e3", c100="#f5e6c8", c200="#e8d5a3", c300="#d4bc7c", c400="#c4a55a", c500="#a08040", c600="#7d6330", c700="#5c4a25", c800="#3b301a", c900="#1a1810", c950="#0d0c08"),
    secondary_hue=gr.themes.Color(c50="#f0f7ee", c100="#d4e8cf", c200="#a8d19f", c300="#7bb870", c400="#5a9e4e", c500="#3d7a35", c600="#2f5f29", c700="#23451e", c800="#172c14", c900="#0b160a", c950="#060b05"),
    neutral_hue=gr.themes.Color(c50="#faf8f5", c100="#f0ece4", c200="#e0d8cc", c300="#c8bfb0", c400="#a89d8c", c500="#887a68", c600="#6b5f4e", c700="#4e4538", c800="#332e24", c900="#1a1712", c950="#0d0c09"),
    font=("Georgia", "serif"),
).set(
    body_background_fill="#faf8f2",
    block_background_fill="#fffdf7",
    block_border_width="1px",
    block_border_color="#e0d8cc",
    block_shadow="0 2px 8px rgba(0,0,0,0.06)",
    block_title_text_color="#4e4538",
    button_primary_background_fill="linear-gradient(135deg, #7bb870, #5a9e4e)",
    button_primary_text_color="white",
    button_primary_border_color="#3d7a35",
)

CSS = """
.gradio-container {max-width: 960px !important; margin: auto !important;}
h1 {text-align: center; color: #4e4538; font-style: italic;}
p {text-align: center; color: #6b5f4e; font-style: italic;}
.fixed-row {flex-wrap: nowrap !important;}
.fixed-row > div {flex: 1 1 50% !important; min-width: 0 !important;}
"""

with gr.Blocks(title="CycleGAN Monet Style Transfer", theme=monet_theme, css=CSS) as demo:
    gr.Markdown("# Impressions of Light  &mdash;  Monet Style Transfer")
    gr.Markdown("*Upload a photograph and let CycleGAN reimagine it in the brushstrokes of Claude Monet.*")

    with gr.Row(equal_height=True, elem_classes="fixed-row"):
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Photograph", height=400, value="images/example.png")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Monet Impression", height=400)

    btn = gr.Button("Paint", variant="primary")
    btn.click(fn=transform_to_monet, inputs=input_image, outputs=output_image)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
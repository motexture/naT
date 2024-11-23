import torch
import numpy as np
import cv2
import os
import gradio as gr

from pipeline.naT import naTPipeline
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

class VideoGenerator:
    def __init__(self):
        self.device = "cuda"
        self.stacked_latents = None
        self.previous_latents = None
        self.video_path = "outputs/output.mp4"
        os.makedirs("outputs", exist_ok=True)

    def set_pipeline(self, model):
        self.pipeline = self.initialize_pipeline(model)

    def initialize_pipeline(self, model):  
        print("Loading pipeline...")
        
        pipeline = naTPipeline.from_pretrained(pretrained_model_name_or_path=model, use_safetensors=False).to(device=self.device, dtype=torch.float16)
        pipeline.vae.enable_slicing()

        return pipeline

    def scale_latents_to_range(self, latents, min_val=-1, max_val=1):
        min_latent = latents.min()
        max_latent = latents.max()
        scaled_latents = (latents - min_latent) / (max_latent - min_latent) * (max_val - min_val) + min_val
        return scaled_latents

    def match_histogram(self, source, target):
        """Adjust the color histogram of the target to match the source."""
        mean_src = source.mean(dim=(2, 3, 4), keepdim=True)
        std_src = source.std(dim=(2, 3, 4), keepdim=True)
        
        mean_tgt = target.mean(dim=(2, 3, 4), keepdim=True)
        std_tgt = target.std(dim=(2, 3, 4), keepdim=True)
        
        adjusted_target = (target - mean_tgt) * (std_src / (std_tgt + 1e-5)) + mean_src
        return adjusted_target

    def generate(self, prompt, negative_prompt, guidance_scale, num_inference_steps, fps):
        with torch.no_grad(), torch.autocast(self.device):
            latents = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=256,
                width=256,
                num_frames=32,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                device=self.device
            )

            self.save_video(self.decode(latents), self.video_path, fps)

            return self.video_path
    
    def decode(self, latents):
        latents = 1 / self.pipeline.vae.config.scaling_factor * latents

        batch, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch * num_frames, channels, height, width)
        image = self.pipeline.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )

        return video.float()  
        
    def denormalize(self, normalized_tensor):
        if normalized_tensor.is_cuda:
            normalized_tensor = normalized_tensor.cpu()
        
        if normalized_tensor.dim() == 5:
            normalized_tensor = normalized_tensor.squeeze(0)
            
        denormalized = (normalized_tensor + 1.0) * 127.5
        denormalized = torch.clamp(denormalized, 0, 255)
        
        uint8_tensor = denormalized.to(torch.uint8)
        uint8_numpy = uint8_tensor.permute(1, 2, 3, 0).numpy()
        
        return uint8_numpy

    def save_video(self, normalized_tensor, output_path, fps=30):
        denormalized_frames = self.denormalize(normalized_tensor)
        height, width = denormalized_frames.shape[1:3]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in denormalized_frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        out.release()      

video_gen = VideoGenerator()

with gr.Blocks() as iface:
    gr.Markdown("""
    <div style="text-align: center;">
        <h1>naT</h1>
        <p>Not Another Text To Video Model</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", value="An astronaut is riding a green horse")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.0, maximum=30.0, step=0.1, value=20.0)

            gr.Markdown("## Inference Settings")

            num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, step=1, value=50)
            fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=16)

        with gr.Column(scale=2):
            video_output = gr.Video(label="Generated Video", width=256, height=256)
            generate_button = gr.Button("Generate Video")

        def on_generate(prompt, negative_prompt, guidance_scale, num_inference_steps, fps):
            global initial_generated
            video_path = video_gen.generate(prompt, negative_prompt, guidance_scale, num_inference_steps, fps)
            initial_generated = True
            return video_path

        generate_button.click(
            on_generate,
            inputs=[prompt, negative_prompt, guidance_scale, num_inference_steps, fps],
            outputs=video_output
        )

if __name__ == "__main__":
    video_gen.set_pipeline("motexture/naT-text-to-video-Alpha-0.1")
    iface.launch()

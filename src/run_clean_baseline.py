from diffusers import DDPMPipeline
import torch
import torchvision.utils as vutils
import os
import numpy as np

pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
pipe.to("cuda")  # or "cpu"

images = pipe(batch_size=16).images  # list of PIL Images

# Convert PIL -> numpy -> torch (C,H,W) in [0,1]
image_tensor = torch.stack([
    torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    for img in images
])

os.makedirs("outputs/clean", exist_ok=True)
vutils.save_image(image_tensor, "outputs/clean/clean_samples.png", nrow=4)
print("Saved outputs/clean/clean_samples.png")
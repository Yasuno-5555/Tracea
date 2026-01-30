
import torch
from safetensors.torch import save_file

# Create dummy Llama weights
tensors_llama = {
    "model.layers.0.self_attn.q_proj.weight": torch.zeros((4096, 4096)),
    "model.embed_tokens.weight": torch.zeros((32000, 4096))
}
save_file(tensors_llama, "llama_dummy.safetensors")
print("Created llama_dummy.safetensors")

# Create dummy Stable Diffusion weights
tensors_sd = {
    "down_blocks.0.resnets.0.conv1.weight": torch.zeros((320, 320, 3, 3)),
    "time_emb_proj.weight": torch.zeros((1280, 320))
}
save_file(tensors_sd, "sd_dummy.safetensors")
print("Created sd_dummy.safetensors")

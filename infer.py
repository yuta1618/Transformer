import argparse
import os
from typing import Optional

import torch
import torch.nn.functional as F

from config import ModelConfig
from src.modules.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-performance Transformer inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a saved checkpoint")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device to run inference on")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Number of tokens to generate")
    parser.add_argument("--prompt_length", type=int, default=4, help="Initial prompt token length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling cutoff")
    parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile for inference speed")
    parser.add_argument("--use_fp16", action="store_true", help="Use mixed precision on CUDA for faster inference")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic sampling")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Optional[str], device: torch.device) -> Transformer:
    config = ModelConfig()
    model = Transformer(config).to(device)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)

    model.eval()
    return model


def generate_tokens(
    model: Transformer,
    context: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    use_fp16: bool,
) -> torch.Tensor:
    idx = context.to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.amp.autocast(device_type="cuda", enabled=use_fp16 and device.type == "cuda"):
                logits, _ = model(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, top_k)
                cutoff = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < cutoff, torch.tensor(-float("inf"), device=device), logits)
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    return idx


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if args.prompt_length <= 0:
        raise ValueError("prompt_length must be positive")

    prompt = torch.randint(0, ModelConfig().vocab_size, (1, args.prompt_length), dtype=torch.long)

    print(f"Running inference on {device} with prompt length={args.prompt_length}, max_new_tokens={args.max_new_tokens}")
    output = generate_tokens(
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        use_fp16=args.use_fp16,
    )

    print("Generated token sequence:")
    print(output.squeeze().tolist())


if __name__ == "__main__":
    main()

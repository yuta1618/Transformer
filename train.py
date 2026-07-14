import argparse
import os
from dataclasses import replace
from typing import Tuple

import torch
import torch.nn.functional as F

from config import ModelConfig
from src.engine import Trainer
from src.modules.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced Transformer training runner")
    parser.add_argument("--max_steps", type=int, help="Total training steps")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Base learning rate")
    parser.add_argument("--resume", type=str, help="Checkpoint path to resume from")
    parser.add_argument("--save_dir", type=str, help="Checkpoint directory")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training on CUDA")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], help="Device to run on")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    return parser.parse_args()


def merge_args(config: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    kwargs = {}
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate
    if args.save_dir is not None:
        kwargs["checkpoint_dir"] = args.save_dir
    if args.resume is not None:
        kwargs["resume_checkpoint"] = args.resume
    if args.fp16:
        kwargs["use_fp16"] = True
    if args.device is not None:
        kwargs["device"] = args.device
    if args.seed is not None:
        kwargs["seed"] = args.seed
    return replace(config, **kwargs)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_toy_dataset(config: ModelConfig) -> torch.Tensor:
    torch.manual_seed(config.seed)
    return torch.randint(0, config.vocab_size, (config.train_dataset_size,), dtype=torch.long)


def get_batch(data: torch.Tensor, config: ModelConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    block_size = config.block_size
    max_start = data.size(0) - block_size - 1
    indices = torch.randint(0, max_start, (config.batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in indices])
    return x, y


def schedule_lr(optimizer: torch.optim.Optimizer, config: ModelConfig, step: int) -> None:
    if step <= config.warmup_steps and config.warmup_steps > 0:
        lr = config.learning_rate * (step / config.warmup_steps)
    else:
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
        lr = config.min_lr + (config.learning_rate - config.min_lr) * cosine_decay.item()
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def sample_tokens(
    model: Transformer,
    context: torch.Tensor,
    steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    idx = context.to(device)
    with torch.no_grad():
        for _ in range(steps):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                cutoff = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < cutoff, torch.tensor(-float("inf"), device=device), logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
    return idx


def main() -> None:
    args = parse_args()
    config = merge_args(ModelConfig(), args)

    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = Transformer(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trainer = Trainer(model, optimizer, device, grad_clip=config.grad_clip)

    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if config.use_fp16 and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    data = build_toy_dataset(config)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if config.resume_checkpoint is not None:
        trainer.load_checkpoint(config.resume_checkpoint)
        print(f"Resumed training from checkpoint {config.resume_checkpoint}")

    print(f"System initialized. Device: {device}")
    print("Starting advanced training run...")

    for step in range(1, config.max_steps + 1):
        xb, yb = get_batch(data, config)
        loss = trainer.step(xb, yb, scaler=scaler, use_fp16=config.use_fp16)
        schedule_lr(optimizer, config, step)

        if step % config.eval_interval == 0 or step == 1:
            eval_loss = trainer.estimate_loss(
                data,
                eval_steps=config.eval_steps,
                batch_size=min(config.batch_size, 32),
            )
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[step {step:04d}/{config.max_steps}] train_loss={loss:.4f} eval_loss={eval_loss:.4f} lr={current_lr:.6f}"
            )

            sample_context = torch.randint(0, config.vocab_size, (1, 1), dtype=torch.long)
            sample_output = sample_tokens(
                model,
                sample_context,
                steps=config.sample_length,
                temperature=config.sample_temperature,
                top_k=config.sample_top_k,
                device=device,
            )
            print(f"Sample tokens: {sample_output.squeeze().tolist()}")

        if step % config.save_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint-step{step}.pt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    final_path = os.path.join(config.checkpoint_dir, "checkpoint-final.pt")
    trainer.save_checkpoint(final_path)
    print(f"Training finished. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()
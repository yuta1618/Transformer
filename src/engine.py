import torch
from typing import Optional

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip: Optional[float] = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip

    def step(
        self,
        xb: torch.Tensor,
        yb: torch.Tensor,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        use_fp16: bool = False,
    ) -> float:
        self.model.train()
        xb, yb = xb.to(self.device), yb.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_fp16 and self.device.type == "cuda"):
            _, loss = self.model(xb, yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def estimate_loss(self, data: torch.Tensor, eval_steps: int = 10, batch_size: int = 16) -> float:
        self.model.eval()
        losses = []
        block_size = self.model.config.block_size
        for _ in range(eval_steps):
            start = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
            xb = torch.stack([data[i : i + block_size] for i in start])
            yb = torch.stack([data[i + 1 : i + 1 + block_size] for i in start])
            _, loss = self.model(xb.to(self.device), yb.to(self.device))
            losses.append(loss.item())
        return float(sum(losses) / len(losses)) if losses else 0.0

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "device": str(self.device),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

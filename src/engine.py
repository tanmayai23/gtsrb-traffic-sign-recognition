"""Per-epoch train and validation loops."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scheduler=None, grad_clip: float = 5.0,
                    epoch: int = 0, total_epochs: int = 0, quiet: bool = False):
    model.train()
    running_loss, correct, seen = 0.0, 0, 0

    desc = f"  epoch {epoch}/{total_epochs} train"
    it = loader if quiet else tqdm(loader, desc=desc, ncols=88, leave=False)

    for images, targets in it:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        # OneCycle is a per-step schedule, so it advances every batch, not epoch.
        if scheduler is not None:
            scheduler.step()

        bs = targets.size(0)
        running_loss += loss.item() * bs
        correct += (outputs.argmax(1) == targets).sum().item()
        seen += bs

        if not quiet:
            it.set_postfix(loss=f"{running_loss / seen:.3f}",
                           acc=f"{correct / seen:.3f}")

    return running_loss / max(seen, 1), correct / max(seen, 1)


@torch.inference_mode()
def validate(model, loader, criterion, device, quiet: bool = False,
             collect: bool = False):
    """Evaluate. With collect=True also returns labels, predictions and probabilities."""
    model.eval()
    running_loss, correct, seen = 0.0, 0, 0
    all_true, all_pred, all_prob = [], [], []

    it = loader if quiet else tqdm(loader, desc="  validate", ncols=88, leave=False)

    for images, targets in it:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        bs = targets.size(0)
        running_loss += loss.item() * bs
        preds = outputs.argmax(1)
        correct += (preds == targets).sum().item()
        seen += bs

        if collect:
            all_true.append(targets.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(torch.softmax(outputs, dim=1).cpu().numpy())

    loss_avg = running_loss / max(seen, 1)
    acc = correct / max(seen, 1)

    if not collect:
        return loss_avg, acc

    return (loss_avg, acc,
            np.concatenate(all_true), np.concatenate(all_pred),
            np.concatenate(all_prob))

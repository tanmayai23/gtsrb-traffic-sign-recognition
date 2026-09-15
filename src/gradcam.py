"""Grad-CAM, implemented directly with autograd hooks.

Gradient-weighted Class Activation Mapping (Selvaraju et al., ICCV 2017):
weight each feature map of the last conv layer by the average gradient of the
target logit with respect to it, sum, and keep the positive part.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import torch

from .plots import plot_gradcam_panel
from .predict import load_for_inference
from .transforms import load_image, preprocess_deterministic, to_tensor
from .utils import configure_threads, ensure_dir


class GradCAM:
    """Hooks a conv layer and produces class activation maps.

    Use as a context manager so the hooks are always detached; leaving them
    attached across repeated calls stacks duplicates and leaks memory.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            # register_full_backward_hook, not the deprecated register_backward_hook,
            # which reports incorrect gradients for multi-input modules.
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        # eval() for BatchNorm running statistics, but gradients must still flow,
        # so this must NOT run under no_grad/inference_mode.
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        logits[0, class_idx].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("hooks did not fire; is the target layer in the graph?")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam[0, 0].cpu().numpy()

        # An all-zero map happens when no positive evidence reaches this layer;
        # the epsilon keeps it black rather than producing NaNs.
        cam = cam - cam.min()
        denom = cam.max()
        cam = cam / denom if denom > 1e-8 else np.zeros_like(cam)

        return cam, class_idx, float(probs[0, class_idx].item())

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def colorize(cam: np.ndarray, size: int) -> np.ndarray:
    """Upsample the CAM and map it through a colour map. Returns uint8 RGB."""
    resized = cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)
    return (cm.jet(resized)[:, :, :3] * 255).astype(np.uint8)


def overlay_cam(image: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)


def run_gradcam(checkpoint: Path, image_path: Path, out_dir: Path,
                target_class: int | None = None, alpha: float = 0.45,
                display_size: int = 256) -> dict:
    image_path, out_dir = Path(image_path), ensure_dir(out_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"no such image: {image_path}")

    configure_threads()
    model, cfg, class_names, norm, device = load_for_inference(checkpoint)

    raw = load_image(image_path)
    proc = preprocess_deterministic(raw, None, cfg.img_size, cfg.use_clahe)
    tensor = to_tensor(proc, norm["mean"], norm["std"]).unsqueeze(0).to(device)

    with GradCAM(model, model.get_last_conv_layer()) as gc:
        cam, class_idx, confidence = gc(tensor, target_class)

    display = cv2.resize(proc, (display_size, display_size),
                         interpolation=cv2.INTER_NEAREST)
    heat = colorize(cam, display_size)
    blended = overlay_cam(display, heat, alpha)

    out_path = out_dir / f"gradcam_{image_path.stem}.png"
    plot_gradcam_panel(
        display, heat, blended,
        f"[{class_idx}] {class_names[class_idx]}  ({confidence:.1%})  "
        f"- CAM is {cam.shape[0]}x{cam.shape[1]}, upsampled",
        out_path,
    )

    print(f"\nimage: {image_path.name}")
    print(f"class: [{class_idx}] {class_names[class_idx]}  ({confidence:.2%})")
    print(f"CAM resolution: {cam.shape[0]}x{cam.shape[1]} (upsampled to {display_size})")
    print(f"saved: {out_path}")

    return {
        "image": str(image_path), "class_id": class_idx,
        "class_name": class_names[class_idx], "confidence": confidence,
        "cam_shape": list(cam.shape), "output": str(out_path),
    }

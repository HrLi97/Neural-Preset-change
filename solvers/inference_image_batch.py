import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import numpy as np
from omegaconf import OmegaConf
import sys
import importlib
from torchvision.transforms.functional import resize
from pathlib import Path
import glob

# Add project paths
project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "solvers"))

def load_image_even_pil(img_pil):
    """Load PIL image, adjust to even dimensions (for network), return tensor and original size."""
    orig_w, orig_h = img_pil.size
    new_w = orig_w - (orig_w % 2)
    new_h = orig_h - (orig_h % 2)
    if new_w == 0:
        new_w = 2
    if new_h == 0:
        new_h = 2
    img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
    tensor = T.ToTensor()(img_resized).unsqueeze(0)
    return tensor, (orig_h, orig_w)

def tensor_to_numpy(tensor):
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def save_image_from_tensor(tensor, save_path, orig_size=None):
    """Save tensor as image. Resize back to orig_size if given."""
    img_np = tensor_to_numpy(tensor)
    if orig_size is not None:
        orig_h, orig_w = orig_size
        # Only resize if current size differs
        if img_np.shape[0] != orig_h or img_np.shape[1] != orig_w:
            img_pil = Image.fromarray(img_np)
            img_resized = img_pil.resize((orig_w, orig_h), Image.LANCZOS)
            img_np = np.array(img_resized)
    Image.fromarray(img_np).save(save_path)

def process_image_folder_with_style(
    ckpt_path,
    content_img_dir,
    style_img_dir,
    output_dir,
    device="cuda"
):
    # === Load config and model ===
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(project_root, "configs/default.yaml")),
        OmegaConf.load(os.path.join(project_root, "configs/env.yaml"))
    )
    model_name = "neural_styler"
    model_cfg = OmegaConf.load(os.path.join(project_root, f"configs/{model_name}.yaml"))
    cfg = OmegaConf.merge(cfg, model_cfg)
    OmegaConf.update(cfg, "model.name", model_name, merge=True)
    cfg.mode = "test"

    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')
    network_class = getattr(network_mod, cfg.model.name)
    net = network_class(cfg)

    solver_mod = importlib.import_module(f'solvers.{cfg.model.name}_{cfg.model.solver}')
    SolverClass = getattr(solver_mod, 'Solver')
    solver = SolverClass.load_from_checkpoint(ckpt_path, net=net, criterion=None, cfg=cfg)
    net = solver.net.eval().to(device)

    # === Get image lists ===
    content_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    style_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']

    content_paths = []
    for ext in content_exts:
        content_paths.extend(glob.glob(os.path.join(content_img_dir, ext)))
    content_paths = sorted(content_paths)

    style_paths = []
    for ext in style_exts:
        style_paths.extend(glob.glob(os.path.join(style_img_dir, ext)))
    style_paths = sorted(style_paths)

    print(f"🔍 Found {len(content_paths)} content images, {len(style_paths)} style images")

    os.makedirs(output_dir, exist_ok=True)

    for content_path in content_paths:
        content_stem = Path(content_path).stem
        content_pil = Image.open(content_path).convert("RGB")
        content_tensor, orig_size = load_image_even_pil(content_pil)
        content_tensor = content_tensor.to(device)
        H, W = content_tensor.shape[-2:]

        for style_path in style_paths:
            style_stem = Path(style_path).stem

            # Skip if names match (avoid self-stylization)
            if content_stem == style_stem:
                print(f"⏭️ Skipping same-name pair: {content_stem} + {style_stem}")
                continue

            output_name = f"{content_stem}_style-{style_stem}.png"
            output_path = os.path.join(output_dir, output_name)

            if os.path.exists(output_path):
                print(f"✅ Exists, skipping: {output_name}")
                continue

            print(f"🎨 Processing: {content_stem} + {style_stem}")

            # Load and resize style image
            style_pil = Image.open(style_path).convert("RGB")
            style_tensor_full, _ = load_image_even_pil(style_pil)
            style_tensor_full = style_tensor_full.to(device)

            # Resize style to content inference size (H, W)
            if style_tensor_full.shape[-2:] != (H, W):
                style_tensor = resize(style_tensor_full, [H, W], antialias=True)
            else:
                style_tensor = style_tensor_full

            # Inference
            with torch.no_grad():
                _, styled = net(content_tensor, style_tensor)  # [1, 3, H, W]

            # Save with original content resolution
            save_image_from_tensor(styled, output_path, orig_size=orig_size)

    print(f"✅ All results saved to: {output_dir}")

def main():
    ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/yuan_b128_e100/251118_112445_neural_styler_v1/last.ckpt"
    content_img_dir = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/bench_mark/content" 
    style_img_dir = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/bench_mark/style"     
    output_dir = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1117/yuan_b128_e100_1119_1v1"

    process_image_folder_with_style(
        ckpt_path=ckpt_path,
        content_img_dir=content_img_dir,
        style_img_dir=style_img_dir,
        output_dir=output_dir,
        device="cuda"
    )

if __name__ == "__main__":
    main()
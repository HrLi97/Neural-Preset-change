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
import argparse

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
    img_np = tensor_to_numpy(tensor)
    if orig_size is not None:
        orig_h, orig_w = orig_size
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
    device="cuda",
    save_grid=False,
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

    # === Get content image paths ===
    content_exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    content_paths = []
    for ext in content_exts:
        content_paths.extend(glob.glob(os.path.join(content_img_dir, ext)))
    content_paths = sorted(content_paths)

    os.makedirs(output_dir, exist_ok=True)

    print(f"🔍 Found {len(content_paths)} content images")

    # cache style r_s to avoid repeated encoder calls when style repeats
    rs_cache = {}

    for content_path in content_paths:
        content_stem = Path(content_path).stem
        # Build expected style path with same stem
        style_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            candidate = os.path.join(style_img_dir, content_stem + ext)
            if os.path.exists(candidate):
                style_path = candidate
                break

        if style_path is None:
            print(f"⚠️ No matching style image for {content_stem}, skipping.")
            continue

        output_name = f"{content_stem}.png"
        output_path = os.path.join(output_dir, output_name)

        if os.path.exists(output_path):
            print(f"✅ Exists, skipping: {output_name}")
            continue

        print(f"🎨 Processing 1:1 pair: {content_stem}")

        # Load content
        content_pil = Image.open(content_path).convert("RGB")
        content_tensor, orig_size = load_image_even_pil(content_pil)
        content_tensor = content_tensor.to(device)
        H, W = content_tensor.shape[-2:]

        # Load style
        style_pil = Image.open(style_path).convert("RGB")
        style_tensor_full, _ = load_image_even_pil(style_pil)
        style_tensor_full = style_tensor_full.to(device)

        # Resize style to content inference size (H, W)
        if style_tensor_full.shape[-2:] != (H, W):
            style_tensor = resize(style_tensor_full, [H, W], antialias=True)
        else:
            style_tensor = style_tensor_full

        # Inference with cached r_s for speed
        with torch.no_grad():
            key = (style_path, H, W)
            if key in rs_cache:
                r_s = rs_cache[key]
                _, styled = net(content=content_tensor, style=style_tensor, r_s=r_s)
            else:
                _, styled = net(content_tensor, style_tensor)
                # Compute and cache r_s from style once (same dims)
                r_s, _ = net.get_r_and_d(style_tensor)
                rs_cache[key] = r_s.detach()

        # Save with original content resolution
        save_image_from_tensor(styled, output_path, orig_size=orig_size)

        if save_grid:
            # also save a side-by-side grid for quick viewing
            grid_name = f"{content_stem}_grid.png"
            grid_path = os.path.join(output_dir, grid_name)
            c_np = tensor_to_numpy(content_tensor)
            s_np = tensor_to_numpy(style_tensor)
            y_np = tensor_to_numpy(styled)
            h = max(c_np.shape[0], s_np.shape[0], y_np.shape[0])
            w_c = c_np.shape[1]; w_s = s_np.shape[1]; w_y = y_np.shape[1]
            pad = lambda img, h: (Image.fromarray(img).resize((img.shape[1], h), Image.LANCZOS))
            cnp = np.array(pad(c_np, h))
            snp = np.array(pad(s_np, h))
            ynp = np.array(pad(y_np, h))
            grid = np.concatenate([cnp, snp, ynp], axis=1)
            cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"✅ All 1:1 results saved to: {output_dir}")

def main():
    '''
    ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/yuan_b128_e100/251118_112445_neural_styler_v1/epoch=0099.ckpt"
    content_img_dir = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/bench_mark/content" 
    style_img_dir = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/bench_mark/style"     
    output_dir = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1117/yuan_b128_e100_99ckpt_1v1"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--content_dir', required=True)
    parser.add_argument('--style_dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save_grid', action='store_true')
    args = parser.parse_args()

    process_image_folder_with_style(
        ckpt_path=args.ckpt,
        content_img_dir=args.content_dir,
        style_img_dir=args.style_dir,
        output_dir=args.out,
        device=args.device,
        save_grid=args.save_grid,
    )

if __name__ == "__main__":
    main()
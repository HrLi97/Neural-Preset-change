import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import numpy as np
from omegaconf import OmegaConf
import sys
import importlib

project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "solvers"))

from neural_styler_v1 import Solver

def load_image_pil_to_tensor(pil_img):
    return T.ToTensor()(pil_img).unsqueeze(0)

def tensor_to_numpy(tensor):
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def extract_patches_and_positions(img_tensor, patch_size=512, overlap=64):
    _, C, H, W = img_tensor.shape
    step = patch_size - overlap
    patches = []
    positions = []
    for top in range(0, H, step):
        for left in range(0, W, step):
            bottom = min(top + patch_size, H)
            right = min(left + patch_size, W)
            patch = img_tensor[:, :, top:bottom, left:right]
            # Pad to patch_size if needed
            pad_h = patch_size - patch.shape[2]
            pad_w = patch_size - patch.shape[3]
            if pad_h > 0 or pad_w > 0:
                patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h))
            patches.append(patch)
            positions.append((top, left, bottom, right))
    return patches, positions, (C, H, W)

def main():
    ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/best.ckpt"
    content_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/content/img/朝阳.png"
    style_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/content/img/昏暗-人.png"
    output_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/output/c/img/朝阳_style_昏暗-人_fullres_patch_v2.png"

    # === Load model ===
    project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
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
    net = solver.net.eval().cuda()

    # === Load images (full resolution) ===
    content_pil = Image.open(content_path).convert("RGB")
    style_pil = Image.open(style_path).convert("RGB")

    content_tensor = load_image_pil_to_tensor(content_pil).cuda()  # [1,3,H,W]
    style_tensor = load_image_pil_to_tensor(style_pil).cuda()      # [1,3,H_s,W_s]

    print(f"Content: {content_pil.size}, Style: {style_pil.size}")

    # === Step 1: Extract global r_s from full style image ===
    with torch.no_grad():
        _, _, r_s = net(content=style_tensor, style=style_tensor)  # r_s: [1, k, k]
    r_s = r_s.detach()

    # === Step 2: Process content in patches ===
    patch_size = 768
    overlap = 0
    patches, positions, (C, H, W) = extract_patches_and_positions(content_tensor, patch_size=patch_size, overlap=overlap)

    output_full = torch.zeros_like(content_tensor)
    weight = torch.zeros((1, 1, H, W), device=content_tensor.device)

    for patch, (top, left, bottom, right) in zip(patches, positions):
        patch = patch.cuda()
        actual_h = bottom - top
        actual_w = right - left

        # Create dummy style (won't be used since r_s is provided)
        dummy_style = torch.zeros_like(patch)  # any tensor of same batch/device

        with torch.no_grad():
            _, styled_patch, _ = net(content=patch, style=dummy_style, r_s=r_s)

        # Crop to actual size (remove padding)
        styled_crop = styled_patch[:, :, :actual_h, :actual_w]

        # Accumulate with overlap weighting
        output_full[:, :, top:bottom, left:right] += styled_crop
        weight[:, :, top:bottom, left:right] += 1

    # Normalize by weight
    output_full = output_full / weight

    # === Save ===
    result = tensor_to_numpy(output_full)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    main()
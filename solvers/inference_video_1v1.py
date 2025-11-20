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
import argparse

project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "solvers"))

def load_image_even_pil(img_pil):
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

def process_video_1v1(ckpt_path, content_video_path, style_image_path, output_video_path, device="cuda", max_frames=None):
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

    cap = cv2.VideoCapture(content_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {content_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    even_w = width - (width % 2)
    even_h = height - (height % 2)
    if even_w == 0:
        even_w = 2
    if even_h == 0:
        even_h = 2

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    style_pil = Image.open(style_image_path).convert("RGB")
    style_tensor_full, _ = load_image_even_pil(style_pil)
    style_tensor_full = style_tensor_full.to(device)
    if style_tensor_full.shape[-2:] != (even_h, even_w):
        style_tensor = resize(style_tensor_full, [even_h, even_w], antialias=True)
    else:
        style_tensor = style_tensor_full

    with torch.no_grad():
        r_s, _ = net.get_r_and_d(style_tensor)
        r_s = r_s.detach()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor, _ = load_image_even_pil(frame_pil)
        frame_tensor = frame_tensor.to(device)
        H, W = frame_tensor.shape[-2:]
        if style_tensor.shape[-2:] != (H, W):
            style_resized = resize(style_tensor, [H, W], antialias=True)
        else:
            style_resized = style_tensor

        with torch.no_grad():
            _, styled = net(content=frame_tensor, style=style_resized, r_s=r_s)
            styled = torch.clamp(styled, 0.0, 1.0)

        styled_np = tensor_to_numpy(styled)
        if (styled_np.shape[1], styled_np.shape[0]) != (width, height):
            styled_np = np.array(Image.fromarray(styled_np).resize((width, height), Image.LANCZOS))
        styled_bgr = cv2.cvtColor(styled_np, cv2.COLOR_RGB2BGR)
        out.write(styled_bgr)
        frame_idx += 1

    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--content_video', required=True)
    parser.add_argument('--style_image', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()

    process_video_1v1(
        ckpt_path=args.ckpt,
        content_video_path=args.content_video,
        style_image_path=args.style_image,
        output_video_path=args.out,
        device=args.device,
        max_frames=args.max_frames,
    )

if __name__ == "__main__":
    main()
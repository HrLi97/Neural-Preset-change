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

# Add project paths
project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "solvers"))

# Import after path setup
from neural_styler_v1 import Solver  # Adjust if solver file name differs

def load_image_keep_ratio_pil(img_pil, max_size=768):
    """Load from PIL Image, keep aspect ratio, max_size limit."""
    orig_w, orig_h = img_pil.size
    scale = min(max_size / orig_h, max_size / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    new_w = new_w - (new_w % 2)
    new_h = new_h - (new_h % 2)
    img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
    tensor = T.ToTensor()(img_resized).unsqueeze(0)
    return tensor, (orig_h, orig_w)

def tensor_to_numpy(tensor):
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * 255, 0, 255).astype(np.uint8)

def process_video_with_style(
    ckpt_path,
    content_video_path,
    style_image_path,
    output_video_path,
    max_size=768,
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

    # === Load style image ===
    style_pil = Image.open(style_image_path).convert("RGB")
    style_tensor_full, _ = load_image_keep_ratio_pil(style_pil, max_size=max_size)
    style_tensor_full = style_tensor_full.to(device)

    # === Open video ===
    cap = cv2.VideoCapture(content_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {content_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: {frame_count} frames, {fps:.2f} FPS, {width}x{height}")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

   # === Step 1: Extract BOTH transform_z (from first frame) and transform_y (from style) ===
   
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Video has no frames!")

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    content_pil = Image.fromarray(first_frame_rgb)
    content_tensor, _ = load_image_keep_ratio_pil(content_pil, max_size=max_size)
    content_tensor = content_tensor.to(device)

    H_infer, W_infer = content_tensor.shape[-2:]
    style_tensor = resize(style_tensor_full, [H_infer, W_infer], antialias=True) \
        if style_tensor_full.shape[-2:] != (H_infer, W_infer) else style_tensor_full
        
    with torch.no_grad():
        r_c, d_c = net.get_r_and_d(content_tensor)  # content features from first frame
        r_s, _   = net.get_r_and_d(style_tensor)    # style features

        transform_z = torch.matmul(torch.matmul(net.transform_p, d_c), net.transform_q)  # [1,3,3]
        transform_y = torch.matmul(torch.matmul(net.transform_p, r_s), net.transform_q)  # [1,3,3]

        transform_z = transform_z.squeeze(0).to(device)  # [3,3]
        transform_y = transform_y.squeeze(0).to(device)  # [3,3]

    print("✅ Extracted transform_z (from first frame) and transform_y (from style).")

    # === Step 2: Apply (transform_y @ transform_z) to all frames ===
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H_orig, W_orig = frame_rgb.shape[:2]
        frame_flat = torch.from_numpy(frame_rgb).to(device).permute(2, 0, 1).reshape(3, -1)

        # Apply whitening THEN coloring
        whitened = torch.matmul(transform_z, frame_flat)      # [3, N]
        styled_flat = torch.matmul(transform_y, whitened)     # [3, N]

        styled_rgb = styled_flat.reshape(3, H_orig, W_orig).permute(1, 2, 0)
        styled_rgb = torch.clamp(styled_rgb, 0, 1).cpu().numpy()
        styled_bgr = cv2.cvtColor((styled_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(styled_bgr)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")

    cap.release()
    out.release()
    print(f"✅ Video saved to: {output_video_path}")


def main():
    ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/best.ckpt"
    content_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/content/video/海洋vid.mp4"
    style_image_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/content/沙漠1.jpg"
    output_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/output/c/海洋vidand沙漠1_onlyFirstFrame.mp4"

    process_video_with_style(
        ckpt_path=ckpt_path,
        content_video_path=content_video_path,
        style_image_path=style_image_path,
        output_video_path=output_video_path,
        max_size=768,  # Adjust based on GPU memory
        device="cuda"
    )

if __name__ == "__main__":
    main()
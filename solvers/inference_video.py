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
# from video_writer import TensorSaveVideo
# from video_stream_writer_ffmpeg import VideoStreamWriter   
from utils_inference.video.video_stream_writer_ffmpeg import VideoStreamWriterFfmpeg

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

def load_image_keep_ratio_pil_content(img_pil):
    """Load from PIL Image, keep aspect ratio, max_size limit."""
    orig_w, orig_h = img_pil.size
    new_w = orig_w - (orig_w % 2)
    new_h = orig_h - (orig_h % 2)
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
    batch_size=1,  # Keep 1 for simplicity; can extend later
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

    # Build model
    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')
    network_class = getattr(network_mod, cfg.model.name)
    net = network_class(cfg)

    solver_mod = importlib.import_module(f'solvers.{cfg.model.name}_{cfg.model.solver}')
    SolverClass = getattr(solver_mod, 'Solver')
    solver = SolverClass.load_from_checkpoint(ckpt_path, net=net, criterion=None, cfg=cfg)
    net = solver.net.eval().to(device)

    # === Load style image (once) ===
    style_pil = Image.open(style_image_path).convert("RGB")
    style_tensor_full, _ = load_image_keep_ratio_pil(style_pil, max_size=4000)
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

    # Setup video writer
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    # if not out.isOpened():
        # raise RuntimeError(f"Failed to open VideoWriter for {output_video_path}")
    
    videoStreamWriterFfmpeg = VideoStreamWriterFfmpeg(save_path=output_video_path, refer_file=content_video_path)
    
    frame_idx = 0
    while frame_idx < 100:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        content_pil = Image.fromarray(frame_rgb)

        # Resize content to match inference size (keep ratio)
        content_tensor, _ = load_image_keep_ratio_pil_content(content_pil)
        content_tensor = content_tensor.to(device)

        # Resize style to match content inference size
        H, W = content_tensor.shape[-2:]
        if style_tensor_full.shape[-2:] != (H, W):
            style_tensor = resize(style_tensor_full, [H, W], antialias=True)
        else:
            style_tensor = style_tensor_full

        # Inference
        with torch.no_grad():
            _, styled = net(content_tensor, style_tensor)  # [1, 3, H, W]
        print(styled.shape,"styledstyledstyled")
        styled_np = tensor_to_numpy(styled)
        style_bytes = np.ascontiguousarray(styled_np).tobytes()
        videoStreamWriterFfmpeg.Write(style_bytes)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")

    cap.release()
    print(f"✅ Video saved to: {output_video_path}")

# def main():
#     ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/best.ckpt"
#     content_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/content/video/大海4k.mp4"
#     style_image_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/content/img/枫树4k.png"
#     output_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/output/c/high_size_2/枫树4k_style-大海4k_2.mp4"
#     os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
#     process_video_with_style(
#         ckpt_path=ckpt_path,
#         content_video_path=content_video_path,
#         style_image_path=style_image_path,
#         output_video_path=output_video_path,
#         max_size=768,
#         device="cuda"
#     )

import glob
from pathlib import Path
def main():
    # ========== 配置根路径 ==========
    base_dir = "/mnt/cfs/shanhai/lihaoran/project/code/color"
    ckpt_path = os.path.join(base_dir, "Neural-Preset-main", "ckps", "best.ckpt")
    video_dir = os.path.join(base_dir, "data", "content", "video")
    style_img_dir = os.path.join(base_dir, "data", "content", "img2")
    output_dir = os.path.join(base_dir, "data", "output", "c", "high_size_3_test")

    os.makedirs(output_dir, exist_ok=True)

    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    style_paths = sorted(glob.glob(os.path.join(style_img_dir, "*")))

    print(f"🔍 找到 {len(video_paths)} 个视频，{len(style_paths)} 个风格图")

    for video_path in video_paths:
        video_stem = Path(video_path).stem  # 如 "朝阳"

        for style_path in style_paths:
            style_stem = Path(style_path).stem  # 如 "昏暗-人" 或 "朝阳"

            if video_stem == style_stem:
                print(f"⏭️ 跳过同名组合: {video_stem} + {style_stem}")
                continue

            output_name = f"{video_stem}_style-{style_stem}-fixed.mp4"
            output_path = os.path.join(output_dir, output_name)

            if os.path.exists(output_path):
                print(f"✅ 已存在，跳过: {output_name}")
                continue

            print(f"🎬 正在处理: {video_stem} + {style_stem}")

            process_video_with_style(
                ckpt_path=ckpt_path,
                content_video_path=video_path,
                style_image_path=style_path,
                output_video_path=output_path,
                max_size=768,
                device="cuda"
            )

if __name__ == "__main__":
    main()
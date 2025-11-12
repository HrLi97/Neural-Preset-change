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
import imageio.v3 as iio
import av
from fractions import Fraction
from utils_inference.video.video_stream_writer_ffmpeg import VideoStreamWriterFfmpeg
from utils_inference.video.video_writer import TensorSaveVideo
from utils_inference.video.video_data import VideoData
from utils_inference.video.video_info_ffmpeg import VideoInfoFfmpeg

project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "solvers"))
from neural_styler_v1 import Solver

def main():
    ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/251107_023648_neural_styler_v1/last.ckpt"
    content_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-未调色.mov"
    style_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-调色.mov"
    output_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-调色-ours_perframe.mov"

    # 加载模型和配置（移到外面，避免重复加载）
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(project_root, "configs/default.yaml")),
        OmegaConf.load(os.path.join(project_root, "configs/env.yaml"))
    )
    model_name = "neural_styler"
    model_cfg = OmegaConf.load(os.path.join(project_root, f"configs/{model_name}.yaml"))
    cfg = OmegaConf.merge(cfg, model_cfg)
    cfg.mode = "test"

    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')
    network_class = getattr(network_mod, cfg.model.name)
    net = network_class(cfg)

    solver_mod = importlib.import_module(f'solvers.{cfg.model.name}_{cfg.model.solver}')
    SolverClass = getattr(solver_mod, 'Solver')
    solver = SolverClass.load_from_checkpoint(ckpt_path, net=net, criterion=None, cfg=cfg)
    net = solver.net.eval().to("cuda").to(torch.float32)

    # 初始化输出流
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    videoStreamWriterFfmpeg = VideoStreamWriterFfmpeg(save_path=output_video_path, refer_file=content_video_path)

    # 使用 VideoInfoFfmpeg 流式读取（避免一次性加载全视频到内存）
    content_info = VideoInfoFfmpeg(content_video_path)
    style_info = VideoInfoFfmpeg(style_video_path)
    total_frames = min(content_info.num_frame, style_info.num_frame,100)
    print(f"Processing {total_frames} paired frames...")

    batch_size = 1  # 逐帧处理，也可以改 batch 提速（需模型支持）
    for start_idx in range(0, total_frames, batch_size):
        end_idx = min(start_idx + batch_size, total_frames)
        indices = list(range(start_idx, end_idx))

        # 加载内容帧
        content_frames = content_info.load_data_by_index(indices)
        content_frames.to_device("cuda").to_nchw().try_norm()

        # 加载风格帧
        style_frames = style_info.load_data_by_index(indices)
        style_frames.to_device("cuda").to_nchw().try_norm()

        with torch.no_grad():
            _, styled = net(content_frames.data, style_frames.data)
            styled = torch.clamp(styled, 0.0, 1.0)

        # 写入输出
        video_data = VideoData(tensor=styled, format=content_frames.format, scale=content_frames.scale)
        video_data.try_denorm()
        TensorSaveVideo.videodata_to_stream(video_data, videoStreamWriterFfmpeg)

        if (start_idx + 1) % 10 == 0:
            print(f"Processed {start_idx + 1}/{total_frames} frames")

    videoStreamWriterFfmpeg.Close()
    print(f"✅ Done. Output: {output_video_path}")

if __name__ == "__main__":
    main()
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

def load_16bit_image_as_tensor(image_path, max_size=4000, device='cpu'):
    img_array = iio.imread(image_path)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    print(f"Style image dtype: {img_array.dtype}, shape: {img_array.shape}, max: {img_array.max()}")

    if img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float32) / 255.0
    elif img_array.dtype == np.uint16:
        max_val = float(img_array.max())
        if max_val == 0:
            norm_factor = 65535.0
        elif max_val <= 1024:
            norm_factor = 1023.0
        else:
            norm_factor = 65535.0
        img_array = img_array.astype(np.float32) / norm_factor
    else:
        img_array = img_array.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / img_array.max()

    tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    h, w = tensor.shape[-2:]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h = int(h * scale) // 2 * 2
        new_w = int(w * scale) // 2 * 2
        tensor = resize(tensor, [new_h, new_w], antialias=True)

    return tensor.unsqueeze(0).to(device), None

def read_10bit_video_with_pyav(content_video_path, max_frames=None):
    container = av.open(content_video_path)
    stream = container.streams.video[0]

    # fps = stream.base_rate
    fps = stream.average_rate  # e.g., Fraction(25, 1)
    if fps is None:
        fps = Fraction(30, 1)
        
    width = stream.width
    height = stream.height
    frame_count = stream.frames if stream.frames > 0 else None
    print(f"Video info: {frame_count if frame_count else '?'} frames, {float(fps):.2f} FPS, {width}x{height}")

    frames = []
    frame_idx = 0
    for frame in container.decode(video=0):
        if max_frames and frame_idx >= max_frames:
            break
        img_array = frame.to_ndarray(format='rgb48')  # uint16, RGB
        frames.append(img_array)
        frame_idx += 1

    container.close()
    return frames, {'fps': fps, 'width': width, 'height': height}

def process_video_with_style(
    ckpt_path,
    content_video_path,
    style_tensor_full,
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
    net.to(torch.float32)

    video_info = VideoInfoFfmpeg(content_video_path)
    # video_info.num_frame
    width, height = video_info.width, video_info.height
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    videoStreamWriterFfmpeg = VideoStreamWriterFfmpeg(save_path=output_video_path, refer_file=content_video_path)
    # for i in range(0,100):
    frames = video_info.load_data_by_index(list(range(100)))
    print(frames.shape,"framesframes")
    
    # for frame in frames:
    for i in range(frames.num_frame):
        frame = frames[i:i+1]
    # offset=0
    # while True:
        # frames=video_info.load_data(offset=offset,max_num_frame=100)
        
        # frames.crop([0,0,width,height])
        
        # if frames is None or frames.num_frame<1:
            # break
        # offset+=frames.num_frame
        # frame.data = frame.data.unsqueeze(0)
        frame.to_device(device).to_nchw().try_norm()
        max_frame_1 = torch.max(frame.data)
        print(max_frame_1,"max_frame_1max_frame_1max_frame_1")
        
        with torch.no_grad():
            _, styled = net(frame.data, style_tensor_full)
            styled = torch.clamp(styled,0.0,1.0)
        # styled = frame.data

        print(styled.shape,"styledstyledstyled")
        max_frame = torch.max(styled)
        print(max_frame,"max_framemax_framemax_framemax_frame") 
        
        video_data = VideoData(tensor=styled,format=frame.format,scale=frame.scale)
        video_data.try_denorm()
        TensorSaveVideo.videodata_to_stream(video_data,videoStreamWriterFfmpeg)

    # for frame_idx, frame in enumerate(frames):
    #     max_val = float(frame.max())
    #     print(max_val,"max_valmax_valmax_valmax_val")
    #     if max_val == 0:
    #         norm_factor = 65535.0
    #     elif max_val <= 1024:
    #         norm_factor = 1023.0
    #     else:
    #         norm_factor = 65535.0
    #     frame_float = frame.astype(np.float32) / norm_factor
    #     content_tensor = torch.from_numpy(frame_float).permute(2, 0, 1).unsqueeze(0).to(device)

    #     # Resize content
    #     orig_h, orig_w = content_tensor.shape[-2:]
    #     scale = min(max_size / orig_h, max_size / orig_w)
    #     new_h = int(orig_h * scale) // 2 * 2
    #     new_w = int(orig_w * scale) // 2 * 2
    #     if (new_h, new_w) != (orig_h, orig_w):
    #         from torch.nn.functional import interpolate
    #         content_tensor = interpolate(content_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

    #     # Resize style
    #     H, W = content_tensor.shape[-2:]
    #     if style_tensor_full.shape[-2:] != (H, W):
    #         style_tensor = interpolate(style_tensor_full, size=(H, W), mode='bilinear', align_corners=False)            
    #     else:
    #         style_tensor = style_tensor_full
    #     style_tensor_full = style_tensor_full.to("cuda")

    #     # Inference
    #     with torch.no_grad():
    #         _, styled = net(content_tensor, style_tensor)

    #     video_data = VideoData(tensor=styled,format="NHWC",scale=2**16-1)
    #     video_data.try_denorm()
    #     TensorSaveVideo.videodata_to_stream(video_data,videoStreamWriterFfmpeg)
        
    #     # styled_np = styled.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    #     # # styled_uint16 = (styled_np * 65535.0).astype(np.uint16)
    #     # styled_uint16 = np.clip(styled_np * 65535.0, 0, 65535).astype(np.uint16)
    #     # style_bytes = np.ascontiguousarray(styled_uint16).tobytes()
    #     # videoStreamWriterFfmpeg.Write(style_bytes)

    # #     # Convert back to uint16 RGB (for 10-bit output)
    # #     styled_np = styled.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, 3] in [0,1]
    # #     styled_uint16 = (styled_np * 65535.0).astype(np.uint16)  # [H, W, 3], uint16

    # #     frame_out = av.VideoFrame.from_ndarray(styled_uint16, format='rgb48le')
    # #     output_stream.pix_fmt = 'yuv422p10le'
    # #     frame_out = frame_out.reformat(width, height, output_stream.pix_fmt)

    # #     # Encode and write
    # #     for packet in output_stream.encode(frame_out):
    # #         output_container.mux(packet)

    # #     if (frame_idx + 1) % 10 == 0:
    # #         print(f"Processed {frame_idx + 1}/{frame_count} frames")

    # # # # Flush encoder
    # # for packet in output_stream.encode():
    # #     output_container.mux(packet)

    # output_container.close()
    videoStreamWriterFfmpeg.Close()
    print(f"✅ Video saved to: {output_video_path}")

def main():
    ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/251107_023648_neural_styler_v1/last.ckpt"
    content_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-未调色.mov"
    output_video_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-调色-ours_0_cpk_ours.mov"  # 保持 .mov 以兼容 ProRes

    # === 从内容视频中提取第一帧作为风格图 ===
    
    style_video_info = VideoInfoFfmpeg("/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-调色.mov")
    style_frame = style_video_info.load_data_by_index([0])
    print(style_frame.data,"style_framestyle_framestyle_frame")
    # frames, _ = read_10bit_video_with_pyav("/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-调色.mov", max_frames=1)
    # style_frame = frames[0]
    # style_tensor_full = style_frame.to_nchw()
    # style_tensor_full = torch.from_numpy(style_frame.astype(np.float32) / 65535.0).permute(2,0,1).unsqueeze(0).to("cuda")  
    # style_frame.data = style_frame.data.unsqueeze(0)
    style_frame.to_device("cuda").to_nchw().try_norm()
    
    process_video_with_style(
        ckpt_path=ckpt_path,
        content_video_path=content_video_path,
        style_tensor_full=style_frame.data,
        output_video_path=output_video_path,
        max_size=768,
        device="cuda"
    )

if __name__ == "__main__":
    main()
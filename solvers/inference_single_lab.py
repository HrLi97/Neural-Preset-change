import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import numpy as np
from omegaconf import OmegaConf
import sys
sys.path.insert(1,os.path.abspath('..'))
sys.path.insert(1,os.path.abspath('../../'))
# from utils.setup import init_path_and_expname,get_callbacks,get_logger,get_trainer_args
import importlib
sys.path.append("/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/solvers/")
project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "solvers"))
from neural_styler_v1 import Solver  
from datasets.rgb2lab import lab_tensor_to_srgb_image, srgb_tensor_to_normalized_lab


def load_image(path, size=512):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    transform = T.Compose([
        T.ToTensor(),  # [0,1]
    ])
    return transform(img).unsqueeze(0)  # add batch dim


def load_image_keep_ratio_content(path):
    img = Image.open(path).convert('RGB')
    orig_w, orig_h = img.size
    new_w = orig_w - (orig_w % 2)
    new_h = orig_h - (orig_h % 2)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    transform = T.ToTensor()
    tensor = transform(img_resized).unsqueeze(0)  # [1, C, H, W]

    return tensor, (orig_h, orig_w)

def load_image_keep_ratio(path, max_size=512):
    img = Image.open(path).convert('RGB')
    orig_w, orig_h = img.size

    scale = min(max_size / orig_h, max_size / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    # 确保尺寸是偶数（避免某些模型下采样出错）
    new_w = new_w - (new_w % 2)
    new_h = new_h - (new_h % 2)

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    transform = T.ToTensor()
    tensor = transform(img_resized).unsqueeze(0)  # [1, C, H, W]

    return tensor, (orig_h, orig_w)

def tensor_to_numpy(tensor):
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def main():
    ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/use_lab_12forlAndAb_andMomentLoss/251114_165757_neural_styler_v1/last.ckpt"
    content_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-未调色_8bit.png"
    style_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1107/喜人课间EP01-P1-调色_8bit.png"
    output_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/output/c/img/喜人课间EP01-frame0_ours_lab_moment_2.png"

    # === 配置加载 ===
    project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
    sys.path.append(project_root)

    # 1. 基础配置
    cfg = OmegaConf.merge(
        OmegaConf.load(os.path.join(project_root, "configs/default.yaml")),
        OmegaConf.load(os.path.join(project_root, "configs/env.yaml"))
    )

    model_name = "neural_styler" 

    model_cfg_path = os.path.join(project_root, f"configs/{model_name}.yaml")
    if not os.path.exists(model_cfg_path):
        raise FileNotFoundError(f"Model config not found: {model_cfg_path}")
    model_cfg = OmegaConf.load(model_cfg_path)
    cfg = OmegaConf.merge(cfg, model_cfg)

    if not hasattr(cfg, 'model') or not hasattr(cfg.model, 'name'):
        OmegaConf.update(cfg, "model.name", model_name, merge=True)

    cfg.mode = "test"
    cfg.seed = 42

    # === 模型初始化 ===
    network_mod = importlib.import_module(f'models.{cfg.model.name}_{cfg.model.ver}')
    network_class = getattr(network_mod, cfg.model.name)
    net = network_class(cfg)

    solver_mod = importlib.import_module(f'solvers.{cfg.model.name}_{cfg.model.solver}')
    SolverClass = getattr(solver_mod, 'Solver')

    solver = SolverClass.load_from_checkpoint(
        ckpt_path,
        net=net,
        criterion=None,
        cfg=cfg
    )
    net = solver.net.eval().cuda()

    def preprocess_image(path, size=None):
        img = Image.open(path).convert('RGB')
        if size is not None:
            img = img.resize(size[::-1], Image.BICUBIC)
        tensor = T.ToTensor()(img) 
        lab = srgb_tensor_to_normalized_lab(tensor)  
        return lab.unsqueeze(0) 

    # Load content and style
    content_img = preprocess_image(content_path, size=(512, 512))
    style_img = preprocess_image(style_path, size=(512, 512))

    # Ensure same size
    if content_img.shape[-2:] != style_img.shape[-2:]:
        from torchvision.transforms.functional import resize
        style_img = resize(style_img, list(content_img.shape[-2:]), antialias=True)

    img_i = content_img.cuda()
    img_j = style_img.cuda()
    
    with torch.no_grad():
        _, styled_output = net(img_i, img_j)  # styled_output is in LAB space

    # Convert LAB to sRGB for visualization
    rgb_image = lab_tensor_to_srgb_image(styled_output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()
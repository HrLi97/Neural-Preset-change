import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import numpy as np
from omegaconf import OmegaConf
import sys
from tqdm import tqdm
import importlib
sys.path.append("/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/solvers/")
project_root = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main"
sys.path.append(project_root)
from datasets.rgb2lab import lab_tensor_to_srgb_image, srgb_tensor_to_normalized_lab

def load_image_keep_ratio_content(path):
    img = Image.open(path).convert('RGB')
    orig_w, orig_h = img.size
    new_w = orig_w - (orig_w % 2)
    new_h = orig_h - (orig_h % 2)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    transform = T.ToTensor()
    tensor = transform(img_resized).unsqueeze(0)
    return tensor, (orig_h, orig_w)

def load_image_keep_ratio(path, max_size=768):
    img = Image.open(path).convert('RGB')
    orig_w, orig_h = img.size
    scale = min(max_size / orig_h, max_size / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    new_w = new_w - (new_w % 2)
    new_h = new_h - (new_h % 2)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    transform = T.ToTensor()
    tensor = transform(img_resized).unsqueeze(0)
    return tensor, (orig_h, orig_w)

def tensor_to_numpy(tensor):
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def load_content_as_lab(path):
    img = Image.open(path).convert('RGB')
    orig_w, orig_h = img.size
    new_w = orig_w - (orig_w % 2)
    new_h = orig_h - (orig_h % 2)
    if (new_w, new_h) != (orig_w, orig_h):
        img = img.resize((new_w, new_h), Image.BICUBIC)
    tensor = T.ToTensor()(img)  # [C, H, W]
    lab = srgb_tensor_to_normalized_lab(tensor)
    return lab.unsqueeze(0), (orig_h, orig_w)

def load_style_as_lab(path, max_size=768):
    img = Image.open(path).convert('RGB')
    orig_w, orig_h = img.size
    scale = min(max_size / orig_h, max_size / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    new_w = new_w - (new_w % 2)
    new_h = new_h - (new_h % 2)
    if (new_w, new_h) != (orig_w, orig_h):
        img = img.resize((new_w, new_h), Image.BICUBIC)
    tensor = T.ToTensor()(img)
    lab = srgb_tensor_to_normalized_lab(tensor)
    return lab.unsqueeze(0)

def main():
    ckpt_path = "/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/ckps/use_lab_12forlAndAb_andMomentLoss/251114_165757_neural_styler_v1/last.ckpt"
    content_dir = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/bench_mark/content"   
    style_dir   = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/bench_mark/style"    
    output_dir  = "/mnt/cfs/shanhai/lihaoran/project/code/color/data/demo/1117/use_lab_12forlAndAb_andMomentLoss_1119/"  

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    content_files = sorted([
        f for f in os.listdir(content_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])
    style_files_set = set([
        f for f in os.listdir(style_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])

    if not content_files:
        raise ValueError(f"No valid images found in content directory: {content_dir}")

    # === 配置加载 ===
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

    # === 创建输出目录 ===
    os.makedirs(output_dir, exist_ok=True)

    # === 遍历 content 图像 ===
    for fname in tqdm(content_files, desc="Processing images"):
        content_path = os.path.join(content_dir, fname)
        style_path = os.path.join(style_dir, fname)

        if fname not in style_files_set:
            print(f"Warning: style image not found for {fname}, skipping.")
            continue

        try:
            content_img,size = load_content_as_lab(content_path)
            style_img = load_style_as_lab(style_path, 768)

            img_i = content_img.cuda()
            img_j = style_img.cuda()

            with torch.no_grad():
                _, styled_output = net(img_i, img_j)  # LAB space, normalized
            rgb_image = lab_tensor_to_srgb_image(styled_output)  # [H, W, 3], uint8, RGB
            output_path = os.path.join(output_dir, fname)
            cv2.imwrite(output_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print(f"All done. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
from PIL import Image
from pillow_lut import load_cube_file
import torch, glob
import os.path as osp
import numpy as np
from omegaconf import OmegaConf
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
import os
import functools
sys.path.append("/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/datasets")
from rgb2lab import srgb_tensor_to_normalized_lab

# class COCO(Dataset):
#     def __init__(self,cfg,split,transform=None):
#         self.cfg = cfg
#         self.root = cfg.data.root
#         self.split = split  # train / valid / test
#         self.lut_root = cfg.data.lut_root

#         self.transform = transform
#         self.totensor = T.ToTensor()
        
#         # prepare data list
#         self.data = []
#         self.data += glob.glob(osp.join(self.root, split, '*.jpg'))

#         # prepare LUTs
#         self.lut = glob.glob(self.lut_root + '/*.cube')
#         self.lut = [load_cube_file(lut) for lut in self.lut]

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # return two type of augmented image, img_i & img_j
#         item_dict = {}
#         img = Image.open(self.data[idx]).convert('RGB')
#         img = img.resize((self.cfg.data.size,self.cfg.data.size),Image.BICUBIC)

#         # get two random LUTs
#         random_lut_idx = np.random.randint(0,len(self.lut),2)
        
#         lut_i = self.lut[random_lut_idx[0]]
#         lut_j = self.lut[random_lut_idx[1]]

#         # apply LUTs
#         img_i = img.filter(lut_i)
#         img_j = img.filter(lut_j)
        
#         if self.transform is not None:
#             img_i = self.transform(img_i)
#             img_j = self.transform(img_j)

#         img_i = self.totensor(img_i)
#         img_j = self.totensor(img_j)
#         # print(img_i,"img_iimg_i")
        
#         item_dict['img_i'] = img_i
#         item_dict['img_j'] = img_j
        
#         print(img_i,"img_j_labimg_j_labimg_j_lab")

#         return item_dict
    
    
class COCO(Dataset):
    def __init__(self,cfg,split,transform=None):
        self.cfg = cfg
        self.root = cfg.data.root
        self.split = split  # train / valid / test
        self.lut_root = cfg.data.lut_root

        self.transform = transform
        self.totensor = T.ToTensor()

        # prepare data list
        self.data = []
        self.data += glob.glob(osp.join(self.root, split, '*.jpg'))

        # prepare LUTs
        # self.lut = glob.glob(self.lut_root + '/*.cube')
        # self.lut = [load_cube_file(lut) for lut in self.lut]
        
        # prepare LUTs path
        self.lut_paths = glob.glob(os.path.join(self.lut_root, "*.cube"))
        self.load_lut = functools.lru_cache(maxsize=48)(self._load_lut_uncached)

    def __len__(self):
        return len(self.data)
    
    def _load_lut_uncached(self, lut_path):
        return load_cube_file(lut_path)

    def __getitem__(self, idx):
        # return two type of augmented image, img_i & img_j
        item_dict = {}
        img = Image.open(self.data[idx]).convert('RGB')
        img = img.resize((self.cfg.data.size,self.cfg.data.size),Image.BICUBIC)

        # # get two random LUTs -yuan
        # random_lut_idx = np.random.randint(0,len(self.lut),2)
        # lut_i = self.lut[random_lut_idx[0]]
        # lut_j = self.lut[random_lut_idx[1]]
        
        # load LUTS
        lut_path_i, lut_path_j = np.random.choice(self.lut_paths, size=2, replace=True)
        lut_i = self.load_lut(lut_path_i)
        lut_j = self.load_lut(lut_path_j)
        
        # apply LUTs
        img_i = img.filter(lut_i)
        img_j = img.filter(lut_j)
        
        if self.transform is not None:
            # TODO ?
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)

        img_i = self.totensor(img_i)
        img_j = self.totensor(img_j)

        # 将SRGB转换到LAB颜色空间;
        img_i_lab = srgb_tensor_to_normalized_lab(img_i)  # [3, H, W], L, a, b
        img_j_lab = srgb_tensor_to_normalized_lab(img_j)
        
        item_dict['img_i'] = img_i_lab
        item_dict['img_j'] = img_j_lab

        return item_dict

def get_loader(cfg,phase):
    # Dataset
    dataset = COCO(cfg=cfg, split=phase)
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader

if __name__=='__main__':
    import sys
    sys.path.append("/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/datasets")
    cfg = OmegaConf.merge(OmegaConf.load('/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/configs/default.yaml'),OmegaConf.load('/mnt/cfs/shanhai/lihaoran/project/code/color/Neural-Preset-main/configs/neural_styler.yaml'),OmegaConf.from_cli())
    test_data = COCO_LAB(
        cfg=cfg,
        split='train'
    )
    test_loader = DataLoader(test_data,batch_size=10,shuffle=True)
    print("The number of data: ",len(test_data))
    print("The number of batch: ",len(test_loader))

    for item_dict in test_loader:
        print(item_dict["img_i"].shape)
        print(item_dict["img_j"].shape)
        input()